import pytest
from pydantic import ValidationError

from inference_core.schemas.llm_config import (
    AllowedOverrideBase,
    NumericConstraints,
    SelectConstraints,
    StringConstraints,
)


class TestConstraintSchemas:
    """Test validation logic for configuration constraints schemas."""

    def test_numeric_constraints_validation(self):
        """Test NumericConstraints validation."""
        # Valid constraint
        c = NumericConstraints(type="number", min=0.0, max=1.0, step=0.1)
        assert c.type == "number"
        assert c.min == 0.0
        assert c.max == 1.0

        # Invalid type literal (should be rejected by Pydantic)
        with pytest.raises(ValidationError):
            NumericConstraints(type="string", min=1.0)

        # Extra fields forbidden
        with pytest.raises(ValidationError):
            NumericConstraints(type="number", extra_field="bad")

    def test_string_constraints_validation(self):
        """Test StringConstraints validation."""
        # Valid constraint
        c = StringConstraints(type="string", pattern="^gpt-", max_length=100)
        assert c.type == "string"
        assert c.pattern == "^gpt-"

        # Invalid type
        with pytest.raises(ValidationError):
            StringConstraints(type="number")

    def test_select_constraints_validation(self):
        """Test SelectConstraints validation."""
        # Valid constraint
        c = SelectConstraints(type="select", allowed_values=["a", "b"])
        assert "a" in c.allowed_values

        # Missing allowed_values (required field)
        with pytest.raises(ValidationError) as excinfo:
            SelectConstraints(type="select")
        assert "allowed_values" in str(excinfo.value)

    def test_discriminated_union(self):
        """Test that ConstraintsUnion correctly discriminates based on 'type' field."""

        # Test Numeric via Union
        numeric_data = {"type": "number", "min": 0, "max": 10}
        model = AllowedOverrideBase(config_key="test", constraints=numeric_data)
        assert isinstance(model.constraints, NumericConstraints)

        # Test String via Union
        string_data = {"type": "string", "pattern": ".*"}
        model = AllowedOverrideBase(config_key="test", constraints=string_data)
        assert isinstance(model.constraints, StringConstraints)

        # Test Invalid Combination (String fields in Number type)
        invalid_data = {"type": "number", "pattern": ".*"}
        with pytest.raises(ValidationError):
            AllowedOverrideBase(config_key="test", constraints=invalid_data)

        # Test Invalid Type
        unknown_type = {"type": "unknown", "min": 1}
        with pytest.raises(ValidationError):
            AllowedOverrideBase(config_key="test", constraints=unknown_type)
