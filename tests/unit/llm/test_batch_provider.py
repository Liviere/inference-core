"""
Unit tests for batch provider base class and registry.

Tests the interface contract using mock provider implementations.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

from app.llm.batch import (
    BaseBatchProvider,
    BatchMode,
    BatchStatus,
    BatchProviderRegistry,
    PreparedSubmission,
    ProviderNotRegisteredError,
    ProviderPermanentError,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
    ProviderTransientError,
    UsageInfo,
    batch_provider_registry,
)


class MockBatchProvider(BaseBatchProvider):
    """Mock provider implementation for testing"""
    
    PROVIDER_NAME = "mock_provider"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.supported_models = ["model1", "model2"]
        self.submitted_batches = {}
        self.should_fail = False
        self.fail_with_transient = False
    
    def supports_model(self, model: str) -> bool:
        return model in self.supported_models
    
    def prepare_payloads(
        self, 
        batch_id: UUID, 
        model: str, 
        mode: BatchMode, 
        requests: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> PreparedSubmission:
        return PreparedSubmission(
            batch_id=batch_id,
            provider_name=self.PROVIDER_NAME,
            model=model,
            mode=mode,
            payloads=requests,
            config=config or {}
        )
    
    async def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        if self.should_fail:
            if self.fail_with_transient:
                raise ProviderTransientError("Temporary submission failure")
            else:
                raise ProviderPermanentError("Permanent submission failure")
        
        provider_batch_id = f"mock_batch_{uuid4().hex[:8]}"
        self.submitted_batches[provider_batch_id] = {
            "status": BatchStatus.QUEUED,
            "submitted_at": datetime.now(),
            "submission": prepared_submission
        }
        
        return ProviderSubmitResult(
            provider_batch_id=provider_batch_id,
            status=BatchStatus.QUEUED,
            raw_status="queued",
            submitted_at=datetime.now()
        )
    
    async def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        if provider_batch_id not in self.submitted_batches:
            raise ProviderPermanentError(f"Batch {provider_batch_id} not found")
        
        batch_data = self.submitted_batches[provider_batch_id]
        return ProviderStatus(
            provider_batch_id=provider_batch_id,
            status=batch_data["status"],
            raw_status=batch_data["status"].value
        )
    
    async def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        if provider_batch_id not in self.submitted_batches:
            raise ProviderPermanentError(f"Batch {provider_batch_id} not found")
        
        # Return mock results
        return [
            ProviderResultRow(
                request_id="req_1",
                status="success",
                response={"result": "mock response 1"},
                usage=UsageInfo(prompt_tokens=10, completion_tokens=15, total_tokens=25)
            ),
            ProviderResultRow(
                request_id="req_2", 
                status="success",
                response={"result": "mock response 2"},
                usage=UsageInfo(prompt_tokens=12, completion_tokens=18, total_tokens=30)
            )
        ]
    
    async def cancel(self, provider_batch_id: str) -> bool:
        if provider_batch_id in self.submitted_batches:
            self.submitted_batches[provider_batch_id]["status"] = BatchStatus.CANCELLED
            return True
        return False


class AnotherMockProvider(BaseBatchProvider):
    """Another mock provider for testing multiple registrations"""
    
    PROVIDER_NAME = "another_provider"
    
    def supports_model(self, model: str) -> bool:
        return model == "special_model"
    
    def prepare_payloads(self, batch_id: UUID, model: str, mode: BatchMode, 
                        requests: List[Dict[str, Any]], config: Dict[str, Any] = None) -> PreparedSubmission:
        return PreparedSubmission(
            batch_id=batch_id, provider_name=self.PROVIDER_NAME,
            model=model, mode=mode, payloads=requests, config=config or {}
        )
    
    async def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        return ProviderSubmitResult(
            provider_batch_id="another_batch_123",
            status=BatchStatus.QUEUED, 
            raw_status="queued",
            submitted_at=datetime.now()
        )
    
    async def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        return ProviderStatus(
            provider_batch_id=provider_batch_id, 
            status=BatchStatus.COMPLETED,
            raw_status="completed"
        )
    
    async def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        return []
    
    async def cancel(self, provider_batch_id: str) -> bool:
        return True


class TestBaseBatchProvider:
    """Test the BaseBatchProvider interface"""
    
    def test_mock_provider_interface(self):
        """Test that mock provider implements the interface correctly"""
        provider = MockBatchProvider()
        
        # Test basic attributes
        assert provider.PROVIDER_NAME == "mock_provider"
        assert hasattr(provider, 'config')
        
        # Test supports_model
        assert provider.supports_model("model1") is True
        assert provider.supports_model("unsupported") is False
    
    def test_prepare_payloads(self):
        """Test prepare_payloads method"""
        provider = MockBatchProvider()
        batch_id = uuid4()
        requests = [{"input": "test"}]
        
        result = provider.prepare_payloads(
            batch_id=batch_id,
            model="model1",
            mode=BatchMode.CHAT,
            requests=requests
        )
        
        assert isinstance(result, PreparedSubmission)
        assert result.batch_id == batch_id
        assert result.provider_name == "mock_provider"
        assert result.model == "model1"
        assert result.mode == BatchMode.CHAT
        assert result.payloads == requests
    
    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Test successful batch submission"""
        provider = MockBatchProvider()
        prepared = PreparedSubmission(
            batch_id=uuid4(),
            provider_name="mock_provider",
            model="model1",
            mode=BatchMode.CHAT,
            payloads=[{"input": "test"}]
        )
        
        result = await provider.submit(prepared)
        
        assert isinstance(result, ProviderSubmitResult)
        assert result.provider_batch_id.startswith("mock_batch_")
        assert result.status == BatchStatus.QUEUED
        assert isinstance(result.submitted_at, datetime)
    
    @pytest.mark.asyncio
    async def test_submit_transient_error(self):
        """Test submission with transient error"""
        provider = MockBatchProvider()
        provider.should_fail = True
        provider.fail_with_transient = True
        
        prepared = PreparedSubmission(
            batch_id=uuid4(), provider_name="mock_provider",
            model="model1", mode=BatchMode.CHAT, payloads=[{"input": "test"}]
        )
        
        with pytest.raises(ProviderTransientError):
            await provider.submit(prepared)
    
    @pytest.mark.asyncio
    async def test_submit_permanent_error(self):
        """Test submission with permanent error"""
        provider = MockBatchProvider()
        provider.should_fail = True
        provider.fail_with_transient = False
        
        prepared = PreparedSubmission(
            batch_id=uuid4(), provider_name="mock_provider",
            model="model1", mode=BatchMode.CHAT, payloads=[{"input": "test"}]
        )
        
        with pytest.raises(ProviderPermanentError):
            await provider.submit(prepared)
    
    @pytest.mark.asyncio
    async def test_poll_status(self):
        """Test polling batch status"""
        provider = MockBatchProvider()
        
        # Submit a batch first
        prepared = PreparedSubmission(
            batch_id=uuid4(), provider_name="mock_provider",
            model="model1", mode=BatchMode.CHAT, payloads=[{"input": "test"}]
        )
        submit_result = await provider.submit(prepared)
        
        # Poll status
        status = await provider.poll_status(submit_result.provider_batch_id)
        
        assert isinstance(status, ProviderStatus)
        assert status.provider_batch_id == submit_result.provider_batch_id
        assert status.status == BatchStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_fetch_results(self):
        """Test fetching batch results"""
        provider = MockBatchProvider()
        
        # Submit a batch first
        prepared = PreparedSubmission(
            batch_id=uuid4(), provider_name="mock_provider",
            model="model1", mode="chat", payloads=[{"input": "test"}]
        )
        submit_result = await provider.submit(prepared)
        
        # Fetch results
        results = await provider.fetch_results(submit_result.provider_batch_id)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ProviderResultRow) for r in results)
        assert results[0].request_id == "req_1"
        assert results[0].status == "success"
    
    @pytest.mark.asyncio
    async def test_cancel(self):
        """Test cancelling a batch"""
        provider = MockBatchProvider()
        
        # Submit a batch first
        prepared = PreparedSubmission(
            batch_id=uuid4(), provider_name="mock_provider",
            model="model1", mode=BatchMode.CHAT, payloads=[{"input": "test"}]
        )
        submit_result = await provider.submit(prepared)
        
        # Cancel the batch
        success = await provider.cancel(submit_result.provider_batch_id)
        
        assert success is True
        
        # Verify status changed
        status = await provider.poll_status(submit_result.provider_batch_id)
        assert status.status == BatchStatus.CANCELLED


class TestBatchProviderRegistry:
    """Test the BatchProviderRegistry functionality"""
    
    def setup_method(self):
        """Setup clean registry for each test"""
        self.registry = BatchProviderRegistry()
    
    def test_register_valid_provider(self):
        """Test registering a valid provider"""
        self.registry.register(MockBatchProvider)
        
        assert self.registry.is_registered("mock_provider")
        assert "mock_provider" in self.registry.list()
    
    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        self.registry.register(MockBatchProvider)
        self.registry.register(AnotherMockProvider)
        
        providers = self.registry.list()
        assert "mock_provider" in providers
        assert "another_provider" in providers
        assert len(providers) == 2
    
    def test_get_registered_provider(self):
        """Test retrieving a registered provider"""
        self.registry.register(MockBatchProvider)
        
        provider_class = self.registry.get("mock_provider")
        assert provider_class == MockBatchProvider
    
    def test_get_unregistered_provider_raises_error(self):
        """Test that accessing unregistered provider raises exception"""
        with pytest.raises(ProviderNotRegisteredError) as exc_info:
            self.registry.get("nonexistent_provider")
        
        assert "nonexistent_provider" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)
    
    def test_register_invalid_provider_class(self):
        """Test registering invalid provider class"""
        class NotAProvider:
            pass
        
        with pytest.raises(ValueError) as exc_info:
            self.registry.register(NotAProvider)
        
        assert "must inherit from BaseBatchProvider" in str(exc_info.value)
    
    def test_register_provider_without_name(self):
        """Test registering provider without PROVIDER_NAME"""
        class ProviderWithoutName(BaseBatchProvider):
            pass
        
        with pytest.raises(ValueError) as exc_info:
            self.registry.register(ProviderWithoutName)
        
        assert "must define a PROVIDER_NAME constant" in str(exc_info.value)
    
    def test_unregister_provider(self):
        """Test unregistering a provider"""
        self.registry.register(MockBatchProvider)
        assert self.registry.is_registered("mock_provider")
        
        success = self.registry.unregister("mock_provider")
        assert success is True
        assert not self.registry.is_registered("mock_provider")
        
        # Unregistering again should return False
        success = self.registry.unregister("mock_provider")
        assert success is False
    
    def test_clear_registry(self):
        """Test clearing all providers"""
        self.registry.register(MockBatchProvider)
        self.registry.register(AnotherMockProvider)
        assert len(self.registry.list()) == 2
        
        self.registry.clear()
        assert len(self.registry.list()) == 0
    
    def test_provider_override_warning(self, caplog):
        """Test that overriding a provider logs a warning"""
        self.registry.register(MockBatchProvider)
        
        # Create another provider with the same name
        class AnotherMockWithSameName(BaseBatchProvider):
            PROVIDER_NAME = "mock_provider"
            
            def supports_model(self, model: str) -> bool:
                return False
            
            def prepare_payloads(self, batch_id, model, mode, requests, config=None):
                pass
            
            async def submit(self, prepared_submission):
                pass
            
            async def poll_status(self, provider_batch_id):
                pass
            
            async def fetch_results(self, provider_batch_id):
                pass
            
            async def cancel(self, provider_batch_id):
                pass
        
        with caplog.at_level("WARNING"):
            self.registry.register(AnotherMockWithSameName)
        
        assert "Overriding existing provider registration" in caplog.text


class TestGlobalRegistry:
    """Test the global registry instance"""
    
    def setup_method(self):
        """Clean up global registry before each test"""
        batch_provider_registry.clear()
    
    def teardown_method(self):
        """Clean up global registry after each test"""
        batch_provider_registry.clear()
    
    def test_global_registry_available(self):
        """Test that global registry is available and functional"""
        batch_provider_registry.register(MockBatchProvider)
        
        assert batch_provider_registry.is_registered("mock_provider")
        provider_class = batch_provider_registry.get("mock_provider")
        assert provider_class == MockBatchProvider