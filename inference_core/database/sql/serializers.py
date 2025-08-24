"""
Database Serialization Utilities

Utilities for converting between database models and external formats
like JSON, with proper handling of UUID and datetime fields.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import Uuid

ModelType = TypeVar("ModelType", bound=DeclarativeBase)


class ModelSerializer:
    """Handles serialization/deserialization of SQLAlchemy models"""

    @staticmethod
    def to_dict(
        model: DeclarativeBase,
        exclude: Optional[set] = None,
        include_relationships: bool = False,
    ) -> Dict[str, Any]:
        """
        Enhanced model to dict conversion with relationship support

        Args:
            model: SQLAlchemy model instance
            exclude: Set of fields to exclude
            include_relationships: Whether to include relationship data

        Returns:
            Dictionary representation
        """
        exclude = exclude or set()
        result = {}

        # Handle columns
        for column in model.__table__.columns:
            if column.name not in exclude:
                value = getattr(model, column.name)
                result[column.name] = ModelSerializer._serialize_value(value)

        # Handle relationships if requested
        if include_relationships:
            for relationship in model.__mapper__.relationships:
                if relationship.key not in exclude:
                    related_value = getattr(model, relationship.key)
                    if related_value is not None:
                        if hasattr(related_value, "__iter__") and not isinstance(
                            related_value, str
                        ):
                            # One-to-many or many-to-many
                            result[relationship.key] = [
                                ModelSerializer.to_dict(
                                    item, exclude={"created_at", "updated_at"}
                                )
                                for item in related_value
                            ]
                        else:
                            # One-to-one or many-to-one
                            result[relationship.key] = ModelSerializer.to_dict(
                                related_value, exclude={"created_at", "updated_at"}
                            )

        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Convert Python values to JSON-serializable formats"""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, uuid.UUID):
            return str(value)
        elif hasattr(value, "__dict__"):
            # Handle complex objects
            return str(value)
        return value

    @staticmethod
    def from_dict(
        model_class: Type[ModelType],
        data: Dict[str, Any],
        exclude: Optional[set] = None,
    ) -> ModelType:
        """
        Create model instance from dictionary with type conversion

        Args:
            model_class: SQLAlchemy model class
            data: Dictionary with values
            exclude: Set of fields to exclude

        Returns:
            New model instance
        """
        exclude = exclude or set()

        # Convert values to appropriate types
        converted_data = ModelSerializer._deserialize_dict(model_class, data)

        # Filter out excluded fields
        filtered_data = {k: v for k, v in converted_data.items() if k not in exclude}

        return model_class(**filtered_data)

    @staticmethod
    def _deserialize_dict(
        model_class: Type[DeclarativeBase], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert dictionary values to appropriate Python types for the model

        Args:
            model_class: SQLAlchemy model class
            data: Dictionary with values to convert

        Returns:
            Dictionary with converted values
        """
        converted_data = data.copy()

        # Get model columns and their types
        for column in model_class.__table__.columns:
            field_name = column.name

            if field_name in converted_data and converted_data[field_name] is not None:
                value = converted_data[field_name]

                # Handle UUID fields
                if isinstance(column.type, Uuid) and isinstance(value, str):
                    try:
                        converted_data[field_name] = uuid.UUID(value)
                    except ValueError:
                        # Invalid UUID string, remove from data
                        del converted_data[field_name]
                        continue

                # Handle datetime fields
                elif hasattr(column.type, "python_type"):
                    if column.type.python_type == datetime and isinstance(value, str):
                        try:
                            # Try ISO format first
                            converted_data[field_name] = datetime.fromisoformat(
                                value.replace("Z", "+00:00")
                            )
                        except ValueError:
                            # Invalid datetime string, remove from data
                            del converted_data[field_name]
                            continue

        return converted_data

    @staticmethod
    def bulk_to_dict(
        models: List[DeclarativeBase],
        exclude: Optional[set] = None,
        include_relationships: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert list of models to list of dictionaries

        Args:
            models: List of SQLAlchemy model instances
            exclude: Set of fields to exclude
            include_relationships: Whether to include relationship data

        Returns:
            List of dictionaries
        """
        return [
            ModelSerializer.to_dict(model, exclude, include_relationships)
            for model in models
        ]


# Convenience functions for backward compatibility
def model_to_dict(
    model: DeclarativeBase,
    exclude: Optional[set] = None,
    include_relationships: bool = False,
) -> Dict[str, Any]:
    """Convenience function for model serialization"""
    return ModelSerializer.to_dict(model, exclude, include_relationships)


def dict_to_model(
    model_class: Type[ModelType], data: Dict[str, Any], exclude: Optional[set] = None
) -> ModelType:
    """Convenience function for model deserialization"""
    return ModelSerializer.from_dict(model_class, data, exclude)
