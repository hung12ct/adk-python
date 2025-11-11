# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import enum

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai.types import Content
from google.genai.types import Part
import pytest


class SessionServiceType(enum.Enum):
  IN_MEMORY = 'IN_MEMORY'
  DATABASE = 'DATABASE'
  SQLITE = 'SQLITE'


def get_session_service(
    service_type: SessionServiceType = SessionServiceType.IN_MEMORY,
    tmp_path=None,
):
  """Creates a session service for testing."""
  if service_type == SessionServiceType.DATABASE:
    return DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  if service_type == SessionServiceType.SQLITE:
    return SqliteSessionService(str(tmp_path / 'sqlite.db'))
  return InMemorySessionService()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type',
    [
        SessionServiceType.IN_MEMORY,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ],
)
async def test_compaction_survives_database_roundtrip(service_type, tmp_path):
  """Test EventCompaction remains an object after DB save/load.

  Reproduces bug where event.actions.compaction becomes dict after loading
  from database, causing AttributeError on attribute access.
  """
  session_service = get_session_service(service_type, tmp_path)

  # Create event with EventCompaction
  compaction = EventCompaction(
      start_timestamp=1.0,
      end_timestamp=2.0,
      compacted_content=Content(
          role='user', parts=[Part(text='Compacted summary')]
      ),
  )
  event = Event(
      author='user',
      actions=EventActions(compaction=compaction),
      invocation_id='test_inv',
  )

  # Save to database
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  await session_service.append_event(session=session, event=event)

  # Load from database (simulates app restart)
  loaded_session = await session_service.get_session(
      app_name='test_app', user_id='test_user', session_id=session.id
  )
  loaded_event = loaded_session.events[0]

  # Critical assertions: compaction should be EventCompaction, not dict
  assert isinstance(loaded_event.actions.compaction, EventCompaction)
  # These would raise AttributeError if compaction was a dict
  assert loaded_event.actions.compaction.start_timestamp == 1.0
  assert loaded_event.actions.compaction.end_timestamp == 2.0
  assert (
      loaded_event.actions.compaction.compacted_content.parts[0].text
      == 'Compacted summary'
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type',
    [
        SessionServiceType.IN_MEMORY,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ],
)
async def test_multiple_events_with_compaction(service_type, tmp_path):
  """Test multiple events with compaction are properly deserialized."""
  session_service = get_session_service(service_type, tmp_path)

  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  # Create and save multiple events
  for i in range(3):
    event = Event(
        author='user',
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=float(i),
                end_timestamp=float(i + 1),
                compacted_content=Content(
                    role='user', parts=[Part(text=f'Summary {i}')]
                ),
            )
        ),
        invocation_id=f'inv_{i}',
    )
    await session_service.append_event(session=session, event=event)

  # Load and verify all
  loaded_session = await session_service.get_session(
      app_name='test_app', user_id='test_user', session_id=session.id
  )

  for i, loaded_event in enumerate(loaded_session.events):
    assert isinstance(loaded_event.actions.compaction, EventCompaction)
    assert loaded_event.actions.compaction.start_timestamp == float(i)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type',
    [
        SessionServiceType.IN_MEMORY,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ],
)
async def test_event_without_compaction(service_type, tmp_path):
  """Test events without compaction are not affected."""
  session_service = get_session_service(service_type, tmp_path)

  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  event = Event(
      author='user', actions=EventActions(), invocation_id='no_compaction'
  )
  await session_service.append_event(session=session, event=event)

  loaded_session = await session_service.get_session(
      app_name='test_app', user_id='test_user', session_id=session.id
  )
  assert loaded_session.events[0].actions.compaction is None
