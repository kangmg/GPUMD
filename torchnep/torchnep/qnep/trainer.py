from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch

from .. import __version__
from .batching import backward_weighted_batch, make_shuffle_generator, shuffled_batches
from .errors import QNEPError
from .gpumd_export import validate_gpumd_export
from .metrics import (
    EpochMetricsRow,
    atomic_save_inference,
    atomic_write_text,
    epoch_metrics_row,
    evaluate_metrics,
    metrics_history_text,
    model_content_hash,
    run_manifest,
    sample_to_model,
    save_best_inference,
    save_best_inference_state,
    validate_output_dir,
)
from .model import QNEPModel
from .records import QNEPDatasetIdentities, validate_datasets
from .run_config import QNEPDatasets, QNEPRunConfig, QNEPTrainingResult
from .training import TrainingConfig
from .training_checkpoint import (
    adam_optimizer_state,
    read_training_checkpoint,
    resolved_run_config,
    restore_training_state,
    save_training_checkpoint,
    validate_resume,
)
from .training_checkpoint_schema import (
    MetricScalar,
    RuntimeMetadata,
    TrainingCheckpointState,
)


def _checkpoint_state(
    model: QNEPModel,
    optimizer: torch.optim.Optimizer,
    config: QNEPRunConfig,
    identities: QNEPDatasetIdentities,
    datasets: QNEPDatasets,
    epoch: int,
    optimizer_step: int,
    best_metric: float,
    best_epoch: int,
    best_model_state: Mapping[str, torch.Tensor],
    patience_count: int,
    initial_model_hash: str,
    history: Sequence[Mapping[str, MetricScalar]],
    shuffle_generator: torch.Generator,
) -> TrainingCheckpointState:
    device = torch.device(config.device)
    device_rng = tuple(torch.cuda.get_rng_state_all()) if device.type == "cuda" else ()
    return TrainingCheckpointState(
        producer_version=__version__,
        model_config=model.config,
        model_state=model.state_dict(),
        optimizer_state=adam_optimizer_state(optimizer),
        completed_epoch=epoch,
        optimizer_step=optimizer_step,
        best_metric=best_metric,
        best_epoch=best_epoch,
        best_model_state=best_model_state,
        patience_count=patience_count,
        train_fingerprint=identities.train,
        validation_fingerprint=identities.validation,
        settings_fingerprint=identities.settings,
        initial_model_hash=initial_model_hash,
        snapshot_id=datasets.snapshot_id,
        metrics_history=tuple(history),
        resolved_config=resolved_run_config(config),
        cpu_rng_state=torch.get_rng_state(),
        device_rng_states=device_rng,
        shuffle_rng_state=shuffle_generator.get_state(),
        runtime=RuntimeMetadata.current(device, config.precision),
    )


def train_qnep(
    model: QNEPModel,
    datasets: QNEPDatasets,
    config: QNEPRunConfig,
) -> QNEPTrainingResult:
    validate_gpumd_export(model)
    validate_output_dir(config)
    identities = validate_datasets(datasets, model, config)
    initial_model_hash = model_content_hash(model.state_dict())
    resume_state = (
        None
        if config.resume_from is None
        else read_training_checkpoint(config.resume_from)
    )
    if resume_state is not None:
        validate_resume(resume_state, config, identities, datasets.snapshot_id)
        if model.config != resume_state.model_config:
            raise QNEPError("resume model config mismatch")
        initial_model_hash = resume_state.initial_model_hash

    device = torch.device(config.device)
    dtype = torch.float32 if config.precision == "float32" else torch.float64
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shuffle_generator = make_shuffle_generator(config.seed)
    if resume_state is None:
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        completed_epoch = 0
        optimizer_step = 0
        best_metric = float("inf")
        best_epoch = 0
        best_model_state: Mapping[str, torch.Tensor] = {}
        patience_count = 0
        history: list[Mapping[str, MetricScalar]] = []
    else:
        restore_training_state(resume_state, model, optimizer, shuffle_generator)
        completed_epoch = resume_state.completed_epoch
        optimizer_step = resume_state.optimizer_step
        best_metric = resume_state.best_metric
        best_epoch = resume_state.best_epoch
        best_model_state = resume_state.best_model_state
        patience_count = resume_state.patience_count
        history = list(resume_state.metrics_history)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = config.output_dir / "run.json"
    metrics_path = config.output_dir / "metrics.jsonl"
    latest_path = config.output_dir / "latest.qnep.pt"
    best_path = config.output_dir / "best.qnep.pt"
    checkpoint_path = config.output_dir / "last.training.pt"
    atomic_write_text(
        run_path,
        run_manifest(config, identities, datasets.snapshot_id, initial_model_hash),
    )
    if resume_state is not None:
        atomic_write_text(metrics_path, metrics_history_text(history))
        save_best_inference_state(model.config, best_model_state, best_path)
        atomic_save_inference(model, latest_path)
        model.export_nep(config.output_dir / "nep_last.txt")

    loss_config = TrainingConfig(
        learning_rate=config.learning_rate,
        energy_weight=config.energy_weight,
        force_weight=config.force_weight,
        charge_weight=config.charge_weight,
    )
    total_weight = sum(record.sample.weight for record in datasets.train)
    early_stopping = (
        config.early_stop_patience is not None
        and patience_count >= config.early_stop_patience
    )
    stop_reason = "early_stopping" if early_stopping else "completed"
    if completed_epoch == config.epochs or early_stopping:
        state = _checkpoint_state(
            model,
            optimizer,
            config,
            identities,
            datasets,
            completed_epoch,
            optimizer_step,
            best_metric,
            best_epoch,
            best_model_state,
            patience_count,
            initial_model_hash,
            history,
            shuffle_generator,
        )
        save_training_checkpoint(state, checkpoint_path)

    epochs = () if early_stopping else range(completed_epoch + 1, config.epochs + 1)
    for epoch in epochs:
        model.train()
        for indices in shuffled_batches(
            len(datasets.train), config.batch_size, shuffle_generator
        ):
            samples = tuple(
                sample_to_model(datasets.train[index].sample, model) for index in indices
            )
            optimizer.zero_grad(set_to_none=True)
            backward_weighted_batch(
                model,
                samples,
                loss_config,
                global_record_count=len(datasets.train),
                global_weight=total_weight,
            )
            optimizer.step()
            optimizer_step += 1
        train_metrics = evaluate_metrics(model, datasets.train, loss_config)
        validation_metrics = evaluate_metrics(model, datasets.validation, loss_config)
        row: EpochMetricsRow = epoch_metrics_row(
            epoch, optimizer_step, train_metrics, validation_metrics
        )
        history.append(row)
        improved = validation_metrics.supervised_objective < best_metric
        if improved:
            best_metric = validation_metrics.supervised_objective
            best_epoch = epoch
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1
        completed_epoch = epoch
        early_stop = (
            config.early_stop_patience is not None
            and patience_count >= config.early_stop_patience
        )
        final_epoch = epoch == config.epochs or early_stop
        checkpoint_boundary = epoch % config.checkpoint_every == 0 or final_epoch
        if checkpoint_boundary:
            state = _checkpoint_state(
                model,
                optimizer,
                config,
                identities,
                datasets,
                epoch,
                optimizer_step,
                best_metric,
                best_epoch,
                best_model_state,
                patience_count,
                initial_model_hash,
                history,
                shuffle_generator,
            )
            save_training_checkpoint(state, checkpoint_path)
            atomic_save_inference(model, latest_path)
            model.export_nep(config.output_dir / "nep_last.txt")
        if improved:
            save_best_inference(model, best_path)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        if early_stop:
            stop_reason = "early_stopping"
            break

    return QNEPTrainingResult(
        completed_epoch=completed_epoch,
        optimizer_step=optimizer_step,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        best_inference_path=best_path,
        latest_inference_path=latest_path,
        training_checkpoint_path=checkpoint_path,
    )
