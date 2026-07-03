from agentic_evalkit.errors import (
    AgenticEvalkitError,
    DatasetAccessDenied,
    DatasetConfigRequired,
    DatasetIntegrityError,
    DatasetLicenseRejected,
    DatasetNotFound,
    DatasetProviderUnavailable,
    DatasetRateLimited,
    DatasetSchemaMismatch,
    DatasetSplitNotFound,
    GraderError,
    IncompatibleRuns,
    ManifestValidationError,
    OfflineCacheMiss,
    PluginCompatibilityError,
    TargetFailure,
    TargetTimeout,
    UnsafeCodeRequired,
)

ALL_SUBCLASSES: tuple[type[AgenticEvalkitError], ...] = (
    DatasetNotFound,
    DatasetConfigRequired,
    DatasetSplitNotFound,
    DatasetAccessDenied,
    DatasetLicenseRejected,
    DatasetIntegrityError,
    DatasetSchemaMismatch,
    DatasetProviderUnavailable,
    UnsafeCodeRequired,
    DatasetRateLimited,
    OfflineCacheMiss,
    PluginCompatibilityError,
    TargetFailure,
    TargetTimeout,
    GraderError,
    IncompatibleRuns,
    ManifestValidationError,
)


def test_str_contains_code_and_message() -> None:
    error = DatasetNotFound(message="dataset 'x' was not found")
    text = str(error)
    assert error.code in text
    assert "dataset 'x' was not found" in text


def test_default_code_is_stable_snake_case_derived_from_class_name() -> None:
    assert DatasetNotFound(message="missing").code == "dataset_not_found"
    assert DatasetConfigRequired(message="need config").code == "dataset_config_required"
    assert DatasetSplitNotFound(message="no split").code == "dataset_split_not_found"
    assert DatasetAccessDenied(message="denied").code == "dataset_access_denied"
    assert DatasetLicenseRejected(message="rejected").code == "dataset_license_rejected"
    assert DatasetIntegrityError(message="corrupt").code == "dataset_integrity_error"
    assert DatasetSchemaMismatch(message="mismatch").code == "dataset_schema_mismatch"
    assert DatasetProviderUnavailable(message="down").code == "dataset_provider_unavailable"
    assert UnsafeCodeRequired(message="unsafe").code == "unsafe_code_required"
    assert DatasetRateLimited(message="slow down").code == "dataset_rate_limited"
    assert OfflineCacheMiss(message="offline").code == "offline_cache_miss"
    assert PluginCompatibilityError(message="incompatible").code == "plugin_compatibility_error"
    assert TargetFailure(message="failed").code == "target_failure"
    assert TargetTimeout(message="timed out").code == "target_timeout"
    assert GraderError(message="grader broke").code == "grader_error"
    assert IncompatibleRuns(message="not comparable").code == "incompatible_runs"
    assert ManifestValidationError(message="bad manifest").code == "manifest_validation_error"


def test_context_values_marked_secret_are_never_serialized_into_str() -> None:
    error = DatasetAccessDenied(
        message="access denied",
        context={
            "dataset_id": "org/private",
            "token": AgenticEvalkitError.secret("hf_super_secret_value"),
        },
    )
    text = str(error)
    assert "org/private" in text
    assert "hf_super_secret_value" not in text


def test_context_is_available_but_secrets_are_redacted_in_repr_too() -> None:
    error = TargetFailure(
        message="target crashed",
        context={"api_key": AgenticEvalkitError.secret("sk-should-not-leak")},
    )
    assert "sk-should-not-leak" not in repr(error)
    assert "sk-should-not-leak" not in str(error)


def test_every_subclass_has_a_unique_stable_code() -> None:
    codes = [cls(message="x").code for cls in ALL_SUBCLASSES]
    assert len(codes) == len(set(codes))


def test_all_subclasses_are_catchable_as_agentic_evalkit_error() -> None:
    for cls in ALL_SUBCLASSES:
        try:
            raise cls(message="boom")
        except AgenticEvalkitError as error:
            assert isinstance(error, cls)
        else:
            raise AssertionError(f"{cls} did not raise")


def test_context_without_secrets_round_trips_into_message() -> None:
    error = DatasetNotFound(message="dataset not found", context={"dataset_id": "openai/gsm8k"})
    assert error.context["dataset_id"] == "openai/gsm8k"
