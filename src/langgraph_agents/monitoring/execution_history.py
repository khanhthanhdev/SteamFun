"""
Agent execution history and replay system.

This module provides comprehensive execution history tracking and replay
capabilities for debugging and analysis purposes.
"""

import json
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import threading
from collections import defaultdict

from .execution_monitor import AgentExecutionTracker, ExecutionEvent, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Complete record of an agent execution for historical analysis."""
    
    record_id: str
    agent_name: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    final_status: ExecutionStatus
    
    # Complete execution data
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    events: List[ExecutionEvent]
    intermediate_states: List[Dict[str, Any]]
    
    # Performance metrics
    total_duration: Optional[float]
    processing_time: float
    step_times: Dict[str, float]
    
    # Error information
    errors: List[str]
    warnings: List[str]
    retry_count: int
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            'record_id': self.record_id,
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'final_status': self.final_status.value,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'events': [event.to_dict() for event in self.events],
            'intermediate_states': self.intermediate_states,
            'total_duration': self.total_duration,
            'processing_time': self.processing_time,
            'step_times': self.step_times,
            'errors': self.errors,
            'warnings': self.warnings,
            'retry_count': self.retry_count,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionRecord':
        """Create record from dictionary."""
        # Convert datetime strings back to datetime objects
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['final_status'] = ExecutionStatus(data['final_status'])
        
        # Convert events back to ExecutionEvent objects
        events = []
        for event_data in data['events']:
            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
            event_data['status'] = ExecutionStatus(event_data['status'])
            events.append(ExecutionEvent(**event_data))
        data['events'] = events
        
        return cls(**data)
    
    def get_duration_breakdown(self) -> Dict[str, float]:
        """Get breakdown of time spent in different phases."""
        if not self.events:
            return {}
        
        status_durations = defaultdict(float)
        current_status = None
        current_start = self.start_time
        
        for event in self.events:
            if event.event_type == "status_update":
                if current_status:
                    duration = (event.timestamp - current_start).total_seconds()
                    status_durations[current_status.value] += duration
                
                current_status = event.status
                current_start = event.timestamp
        
        # Handle final status
        if current_status and self.end_time:
            duration = (self.end_time - current_start).total_seconds()
            status_durations[current_status.value] += duration
        
        return dict(status_durations)
    
    def get_error_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of errors during execution."""
        error_events = [
            event for event in self.events 
            if event.event_type == "error" or event.error
        ]
        
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'error': event.error or event.message,
                'context': event.data
            }
            for event in error_events
        ]


@dataclass
class ExecutionReplay:
    """Represents a replay of an execution for debugging."""
    
    original_record: ExecutionRecord
    replay_id: str
    replay_timestamp: datetime = field(default_factory=datetime.now)
    
    # Replay configuration
    replay_speed: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
    stop_on_errors: bool = False
    breakpoints: List[str] = field(default_factory=list)  # Event types to break on
    
    # Replay state
    current_event_index: int = 0
    is_playing: bool = False
    is_paused: bool = False
    
    def get_current_event(self) -> Optional[ExecutionEvent]:
        """Get the current event in the replay."""
        if 0 <= self.current_event_index < len(self.original_record.events):
            return self.original_record.events[self.current_event_index]
        return None
    
    def get_progress(self) -> float:
        """Get replay progress as percentage."""
        if not self.original_record.events:
            return 100.0
        return (self.current_event_index / len(self.original_record.events)) * 100.0
    
    def get_remaining_events(self) -> List[ExecutionEvent]:
        """Get remaining events in the replay."""
        return self.original_record.events[self.current_event_index:]


class ExecutionHistory:
    """
    Comprehensive execution history tracking and replay system.
    
    Provides storage, retrieval, and replay capabilities for agent execution
    records with support for filtering, searching, and analysis.
    """
    
    def __init__(self, storage_path: str = "execution_history", max_records: int = 10000):
        """
        Initialize execution history system.
        
        Args:
            storage_path: Path to store execution records
            max_records: Maximum number of records to keep in memory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        
        # In-memory storage for recent records
        self.recent_records: Dict[str, ExecutionRecord] = {}
        self.agent_records: Dict[str, List[str]] = defaultdict(list)  # agent_name -> record_ids
        
        # Active replays
        self.active_replays: Dict[str, ExecutionReplay] = {}
        
        # Background storage thread
        self._storage_queue: List[ExecutionRecord] = []
        self._storage_lock = threading.Lock()
        self._storage_thread: Optional[threading.Thread] = None
        self._stop_storage = threading.Event()
        
        # Load recent records from disk
        self._load_recent_records()
        
        # Start background storage
        self._start_storage_thread()
        
        logger.info(f"ExecutionHistory initialized with storage at {self.storage_path}")
    
    def record_execution(self, tracker: AgentExecutionTracker) -> ExecutionRecord:
        """Record a completed agent execution."""
        record = ExecutionRecord(
            record_id=f"{tracker.agent_name}_{tracker.session_id}_{int(datetime.now().timestamp())}",
            agent_name=tracker.agent_name,
            session_id=tracker.session_id,
            start_time=tracker.start_time,
            end_time=tracker.end_time,
            final_status=tracker.current_status,
            input_data=tracker.input_data,
            output_data=tracker.output_data,
            events=tracker.events.copy(),
            intermediate_states=tracker.intermediate_states.copy(),
            total_duration=tracker.get_execution_duration(),
            processing_time=tracker.total_processing_time,
            step_times=tracker.step_times.copy(),
            errors=tracker.errors.copy(),
            warnings=tracker.warnings.copy(),
            retry_count=tracker.retry_count
        )
        
        # Add to in-memory storage
        self.recent_records[record.record_id] = record
        self.agent_records[record.agent_name].append(record.record_id)
        
        # Queue for disk storage
        with self._storage_lock:
            self._storage_queue.append(record)
        
        # Cleanup old records if needed
        self._cleanup_old_records()
        
        logger.info(f"Recorded execution: {record.record_id}")
        return record
    
    def get_record(self, record_id: str) -> Optional[ExecutionRecord]:
        """Get a specific execution record."""
        # Check in-memory first
        if record_id in self.recent_records:
            return self.recent_records[record_id]
        
        # Try loading from disk
        return self._load_record_from_disk(record_id)
    
    def get_agent_records(self, agent_name: str, limit: int = 50) -> List[ExecutionRecord]:
        """Get execution records for a specific agent."""
        if agent_name not in self.agent_records:
            return []
        
        record_ids = self.agent_records[agent_name][-limit:]
        records = []
        
        for record_id in record_ids:
            record = self.get_record(record_id)
            if record:
                records.append(record)
        
        return sorted(records, key=lambda r: r.start_time, reverse=True)
    
    def search_records(self, 
                      agent_name: Optional[str] = None,
                      status: Optional[ExecutionStatus] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      has_errors: Optional[bool] = None,
                      tags: Optional[List[str]] = None,
                      limit: int = 100) -> List[ExecutionRecord]:
        """Search execution records with filters."""
        results = []
        
        # Search in recent records
        for record in self.recent_records.values():
            if self._matches_filters(record, agent_name, status, start_date, 
                                   end_date, has_errors, tags):
                results.append(record)
        
        # Sort by start time (most recent first)
        results.sort(key=lambda r: r.start_time, reverse=True)
        
        return results[:limit]
    
    def get_execution_statistics(self, agent_name: Optional[str] = None,
                               days: int = 7) -> Dict[str, Any]:
        """Get execution statistics for analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get relevant records
        if agent_name:
            records = [r for r in self.get_agent_records(agent_name) 
                      if r.start_time >= cutoff_date]
        else:
            records = [r for r in self.recent_records.values() 
                      if r.start_time >= cutoff_date]
        
        if not records:
            return {'total_executions': 0}
        
        # Calculate statistics
        total_executions = len(records)
        successful_executions = len([r for r in records if r.final_status == ExecutionStatus.COMPLETED])
        failed_executions = len([r for r in records if r.final_status == ExecutionStatus.FAILED])
        
        durations = [r.total_duration for r in records if r.total_duration]
        processing_times = [r.processing_time for r in records]
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'total_errors': sum(len(r.errors) for r in records),
            'total_retries': sum(r.retry_count for r in records),
            'agents_involved': len(set(r.agent_name for r in records))
        }
    
    def create_replay(self, record_id: str, replay_config: Optional[Dict[str, Any]] = None) -> Optional[ExecutionReplay]:
        """Create a replay session for an execution record."""
        record = self.get_record(record_id)
        if not record:
            logger.error(f"Cannot create replay: record {record_id} not found")
            return None
        
        config = replay_config or {}
        replay = ExecutionReplay(
            original_record=record,
            replay_id=f"replay_{record_id}_{int(datetime.now().timestamp())}",
            replay_speed=config.get('replay_speed', 1.0),
            stop_on_errors=config.get('stop_on_errors', False),
            breakpoints=config.get('breakpoints', [])
        )
        
        self.active_replays[replay.replay_id] = replay
        logger.info(f"Created replay session: {replay.replay_id}")
        return replay
    
    def control_replay(self, replay_id: str, action: str) -> bool:
        """Control replay playback (play, pause, stop, step)."""
        if replay_id not in self.active_replays:
            return False
        
        replay = self.active_replays[replay_id]
        
        if action == "play":
            replay.is_playing = True
            replay.is_paused = False
        elif action == "pause":
            replay.is_paused = True
        elif action == "stop":
            replay.is_playing = False
            replay.is_paused = False
            replay.current_event_index = 0
        elif action == "step":
            if replay.current_event_index < len(replay.original_record.events):
                replay.current_event_index += 1
        elif action == "reset":
            replay.current_event_index = 0
            replay.is_playing = False
            replay.is_paused = False
        
        return True
    
    def get_replay_status(self, replay_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a replay session."""
        if replay_id not in self.active_replays:
            return None
        
        replay = self.active_replays[replay_id]
        current_event = replay.get_current_event()
        
        return {
            'replay_id': replay_id,
            'original_record_id': replay.original_record.record_id,
            'progress': replay.get_progress(),
            'current_event_index': replay.current_event_index,
            'total_events': len(replay.original_record.events),
            'is_playing': replay.is_playing,
            'is_paused': replay.is_paused,
            'current_event': current_event.to_dict() if current_event else None,
            'replay_speed': replay.replay_speed
        }
    
    def export_records(self, record_ids: List[str], format: str = "json") -> str:
        """Export execution records to file."""
        records = []
        for record_id in record_ids:
            record = self.get_record(record_id)
            if record:
                records.append(record.to_dict())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"execution_export_{timestamp}.json"
            filepath = self.storage_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(records, f, indent=2, default=str)
        
        elif format == "pickle":
            filename = f"execution_export_{timestamp}.pkl"
            filepath = self.storage_path / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(records, f)
        
        logger.info(f"Exported {len(records)} records to {filepath}")
        return str(filepath)
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old execution records."""
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        # Remove from in-memory storage
        to_remove = []
        for record_id, record in self.recent_records.items():
            if record.start_time < cutoff_date:
                to_remove.append(record_id)
        
        for record_id in to_remove:
            del self.recent_records[record_id]
            removed_count += 1
            
            # Remove from agent records
            for agent_name, record_ids in self.agent_records.items():
                if record_id in record_ids:
                    record_ids.remove(record_id)
        
        # Clean up disk storage
        for file_path in self.storage_path.glob("*.json.gz"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old execution records")
        return removed_count
    
    def shutdown(self) -> None:
        """Shutdown the execution history system."""
        # Stop storage thread
        self._stop_storage.set()
        if self._storage_thread:
            self._storage_thread.join(timeout=5.0)
        
        # Save any remaining records
        with self._storage_lock:
            for record in self._storage_queue:
                self._save_record_to_disk(record)
        
        logger.info("ExecutionHistory shutdown complete")
    
    def _matches_filters(self, record: ExecutionRecord,
                        agent_name: Optional[str] = None,
                        status: Optional[ExecutionStatus] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        has_errors: Optional[bool] = None,
                        tags: Optional[List[str]] = None) -> bool:
        """Check if record matches search filters."""
        if agent_name and record.agent_name != agent_name:
            return False
        
        if status and record.final_status != status:
            return False
        
        if start_date and record.start_time < start_date:
            return False
        
        if end_date and record.start_time > end_date:
            return False
        
        if has_errors is not None:
            record_has_errors = len(record.errors) > 0
            if has_errors != record_has_errors:
                return False
        
        if tags:
            if not any(tag in record.tags for tag in tags):
                return False
        
        return True
    
    def _cleanup_old_records(self) -> None:
        """Clean up old records from memory if limit exceeded."""
        if len(self.recent_records) <= self.max_records:
            return
        
        # Sort by start time and keep most recent
        sorted_records = sorted(
            self.recent_records.items(),
            key=lambda x: x[1].start_time,
            reverse=True
        )
        
        # Keep only the most recent records
        to_keep = dict(sorted_records[:self.max_records])
        to_remove = set(self.recent_records.keys()) - set(to_keep.keys())
        
        for record_id in to_remove:
            del self.recent_records[record_id]
            
            # Remove from agent records
            for agent_name, record_ids in self.agent_records.items():
                if record_id in record_ids:
                    record_ids.remove(record_id)
    
    def _start_storage_thread(self) -> None:
        """Start background thread for disk storage."""
        self._storage_thread = threading.Thread(
            target=self._storage_loop,
            daemon=True
        )
        self._storage_thread.start()
    
    def _storage_loop(self) -> None:
        """Background loop for saving records to disk."""
        while not self._stop_storage.is_set():
            try:
                records_to_save = []
                
                with self._storage_lock:
                    if self._storage_queue:
                        records_to_save = self._storage_queue.copy()
                        self._storage_queue.clear()
                
                for record in records_to_save:
                    self._save_record_to_disk(record)
                
                # Wait before next iteration
                self._stop_storage.wait(10)  # Save every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in storage loop: {e}")
                self._stop_storage.wait(10)
    
    def _save_record_to_disk(self, record: ExecutionRecord) -> None:
        """Save a single record to disk."""
        try:
            filename = f"{record.record_id}.json.gz"
            filepath = self.storage_path / filename
            
            with gzip.open(filepath, 'wt') as f:
                json.dump(record.to_dict(), f, default=str)
                
        except Exception as e:
            logger.error(f"Error saving record {record.record_id}: {e}")
    
    def _load_record_from_disk(self, record_id: str) -> Optional[ExecutionRecord]:
        """Load a record from disk storage."""
        try:
            filename = f"{record_id}.json.gz"
            filepath = self.storage_path / filename
            
            if not filepath.exists():
                return None
            
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
                return ExecutionRecord.from_dict(data)
                
        except Exception as e:
            logger.error(f"Error loading record {record_id}: {e}")
            return None
    
    def _load_recent_records(self) -> None:
        """Load recent records from disk on startup."""
        try:
            # Load the most recent records
            record_files = sorted(
                self.storage_path.glob("*.json.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            loaded_count = 0
            for file_path in record_files[:self.max_records]:
                try:
                    with gzip.open(file_path, 'rt') as f:
                        data = json.load(f)
                        record = ExecutionRecord.from_dict(data)
                        
                        self.recent_records[record.record_id] = record
                        self.agent_records[record.agent_name].append(record.record_id)
                        loaded_count += 1
                        
                except Exception as e:
                    logger.error(f"Error loading record from {file_path}: {e}")
            
            logger.info(f"Loaded {loaded_count} execution records from disk")
            
        except Exception as e:
            logger.error(f"Error loading recent records: {e}")


# Global instance
_global_execution_history = ExecutionHistory()


def get_execution_history() -> ExecutionHistory:
    """Get the global execution history instance."""
    return _global_execution_history