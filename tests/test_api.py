# tests/test_api.py - API endpoint testing
import pytest
import asyncio
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import websockets
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.claude_interface import ClaudeConsciousnessInterface, ConversationContext
from src.api.websocket_server import ConsciousnessWebSocketServer
from src.ui.web_interface import AIMSWebInterface

class TestClaudeInterface:
    """Test Claude API integration"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'consciousness': {
                'cycle_frequency': 2.0,
                'working_memory_size': 7
            },
            'memory': {
                'consolidation_threshold': 0.3
            }
        }
    
    @pytest.fixture
    def claude_interface(self, mock_config):
        with patch('anthropic.AsyncAnthropic'):
            interface = ClaudeConsciousnessInterface('test_api_key', mock_config)
            # Mock the subsystems
            interface.consciousness = MagicMock()
            interface.memory_manager = MagicMock()
            interface.personality = MagicMock()
            interface.emotions = MagicMock()
            return interface
    
    @pytest.mark.asyncio
    async def test_initialize_session(self, claude_interface):
        """Test session initialization"""
        # Mock memory retrieval
        claude_interface.memory_manager.retrieve_memories = AsyncMock(return_value=[])
        
        # Initialize session
        context = await claude_interface.initialize_session('test_session', 'test_user')
        
        assert isinstance(context, ConversationContext)
        assert context.session_id == 'test_session'
        assert context.user_id == 'test_user'
        assert context.recent_memories == []
    
    @pytest.mark.asyncio
    async def test_process_message(self, claude_interface):
        """Test message processing"""
        # Setup mocks
        claude_interface.active_sessions['test_session'] = MagicMock(
            user_id='test_user',
            consciousness_state=MagicMock(interaction_count=0)
        )
        
        claude_interface.memory_manager.retrieve_memories = AsyncMock(return_value=[])
        claude_interface.memory_manager.save_conversation_turn = AsyncMock()
        claude_interface.memory_manager.get_conversation_history = AsyncMock(return_value=[])
        
        # Mock Claude API response
        mock_stream = AsyncMock()
        mock_stream.text_stream = AsyncMock()
        mock_stream.text_stream.__aiter__.return_value = ['Hello', ' World']
        
        claude_interface.client.messages.stream = AsyncMock(return_value=mock_stream)
        claude_interface.client.messages.stream().__aenter__ = AsyncMock(return_value=mock_stream)
        claude_interface.client.messages.stream().__aexit__ = AsyncMock()
        
        # Process message
        response_parts = []
        async for chunk in claude_interface.process_message('test_session', 'Hello'):
            response_parts.append(chunk)
        
        assert ''.join(response_parts) == 'Hello World'
    
    @pytest.mark.asyncio
    async def test_analyze_message(self, claude_interface):
        """Test message analysis"""
        # Test positive sentiment
        analysis = await claude_interface._analyze_message("Thank you, this is great!")
        assert analysis['sentiment'] > 0.5
        
        # Test negative sentiment
        analysis = await claude_interface._analyze_message("This is terrible, I hate it")
        assert analysis['sentiment'] < 0.5
        
        # Test urgency
        analysis = await claude_interface._analyze_message("I need this urgently!")
        assert analysis['urgency'] > 0.5

class TestWebSocketServer:
    """Test WebSocket server functionality"""
    
    @pytest.fixture
    def mock_claude_interface(self):
        interface = MagicMock()
        interface.consciousness = MagicMock()
        interface.consciousness.state = MagicMock(
            global_coherence=0.8,
            attention_focus="test",
            interaction_count=0
        )
        interface.consciousness.memory_buffer = []
        interface.emotions = MagicMock()
        interface.emotions.current_state = MagicMock(
            pleasure=0.5,
            arousal=0.5,
            dominance=0.5
        )
        interface.emotions.get_closest_emotion_label = MagicMock(
            return_value=("neutral", 0.8)
        )
        interface.personality = MagicMock()
        interface.personality.profile = MagicMock()
        interface.personality.profile.get_traits = MagicMock(
            return_value={'openness': 0.8}
        )
        interface.memory_manager = MagicMock()
        interface.memory_manager.get_statistics = MagicMock(
            return_value={'total_memories': 0}
        )
        return interface
    
    @pytest.mark.asyncio
    async def test_client_registration(self, mock_claude_interface):
        """Test client registration and reconnection"""
        server = ConsciousnessWebSocketServer(mock_claude_interface)
        
        # Mock websocket
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.remote_address = ('127.0.0.1', 12345)
        
        # Register new client
        client_id = await server.register_client(mock_ws)
        assert client_id in server.connections
        assert server.connections[client_id].reconnect_count == 0
        
        # Test reconnection
        client_id2 = await server.register_client(mock_ws, client_id)
        assert client_id == client_id2
        assert server.connections[client_id].reconnect_count == 1
    
    @pytest.mark.asyncio
    async def test_message_queue(self, mock_claude_interface):
        """Test message queuing for offline clients"""
        server = ConsciousnessWebSocketServer(mock_claude_interface)
        
        # Add message to queue
        test_message = {'type': 'test', 'data': 'hello'}
        server.message_queue.add_message('test_client', test_message)
        
        # Retrieve messages
        messages = server.message_queue.get_messages('test_client')
        assert len(messages) == 1
        assert messages[0]['type'] == 'test'
        assert messages[0]['data'] == 'hello'
        
        # Queue should be empty after retrieval
        messages2 = server.message_queue.get_messages('test_client')
        assert len(messages2) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_claude_interface):
        """Test rate limiting functionality"""
        server = ConsciousnessWebSocketServer(mock_claude_interface)
        server.max_messages_per_minute = 5  # Low limit for testing
        
        client_id = 'test_client'
        
        # Should allow first 5 messages
        for i in range(5):
            assert await server.check_rate_limit(client_id)
        
        # 6th message should be rate limited
        assert not await server.check_rate_limit(client_id)

class TestWebInterface(AioHTTPTestCase):
    """Test web interface endpoints"""
    
    async def get_application(self):
        """Create application for testing"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            interface = AIMSWebInterface()
            # Mock Claude interface
            interface.claude_interface = MagicMock()
            interface.claude_interface.shutdown = AsyncMock()
            return interface.app
    
    @unittest_run_loop
    async def test_index_page(self):
        """Test main page loads"""
        resp = await self.client.request("GET", "/")
        assert resp.status == 200
        text = await resp.text()
        assert "AIMS" in text
    
    @unittest_run_loop
    async def test_chat_endpoint(self):
        """Test chat API endpoint"""
        # Mock the process_message method
        self.app['aims_interface'] = MagicMock()
        
        async def mock_process_message(session_id, message, stream):
            yield "Test response"
        
        with patch.object(
            self.app['aims_interface'].claude_interface,
            'process_message',
            mock_process_message
        ):
            resp = await self.client.request(
                "POST",
                "/api/chat",
                json={"message": "Hello"}
            )
            assert resp.status == 200
            data = await resp.json()
            assert 'response' in data
    
    @unittest_run_loop
    async def test_consciousness_state_endpoint(self):
        """Test consciousness state endpoint"""
        mock_state = {
            'coherence': 0.8,
            'attention': 'test',
            'emotion': {'pleasure': 0.5}
        }
        
        self.app['aims_interface'].claude_interface.consciousness.get_state_summary = MagicMock(
            return_value=mock_state
        )
        self.app['aims_interface'].claude_interface.emotions.get_closest_emotion_label = MagicMock(
            return_value=('neutral', 0.8)
        )
        self.app['aims_interface'].claude_interface.emotions.get_emotional_color = MagicMock(
            return_value=(128, 128, 128)
        )
        
        resp = await self.client.request("GET", "/api/consciousness/state")
        assert resp.status == 200
        data = await resp.json()
        assert data['coherence'] == 0.8
    
    @unittest_run_loop
    async def test_memory_stats_endpoint(self):
        """Test memory statistics endpoint"""
        mock_stats = {
            'total_memories': 100,
            'average_importance': 0.7
        }
        
        self.app['aims_interface'].claude_interface.memory_manager.get_statistics = MagicMock(
            return_value=mock_stats
        )
        
        resp = await self.client.request("GET", "/api/memory/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data['total_memories'] == 100

class TestAuthentication:
    """Test authentication and session management"""
    
    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test that sessions are created properly"""
        # This would test session middleware
        pass
    
    @pytest.mark.asyncio
    async def test_user_isolation(self):
        """Test that users cannot access other users' data"""
        # This would test multi-user support
        pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])