"""Exercise admitted HTTP runs and actual restricted invocation on executor threads."""
import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from agent.jentera_lane_policy import POLICY, POLICY_HASH, apply_policy
from run_agent import AIAgent

@pytest.mark.asyncio
async def test_two_policy_runs_overlap_and_echo_target_proof(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    barrier = threading.Barrier(2, timeout=5)
    intervals = []
    def create(**kwargs):
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        agent._pending_steer = None
        agent._tool_worker_threads = set()
        agent._jentera_tool_futures = set()
        agent._tool_worker_threads_lock = threading.Lock()
        policy = kwargs['jentera_lane_policy']
        def run(**run_kwargs):
            actual = AIAgent.__new__(AIAgent)
            actual.jentera_lane_policy = policy
            began = time.monotonic()
            barrier.wait()
            kwargs['stream_delta_callback']('independent progress')
            assert 'error' in json.loads(actual._invoke_tool('terminal', {'command':'false'}, run_kwargs['task_id']))
            assert 'error' in json.loads(actual._invoke_tool('read_file', {'path':'/etc/passwd'}, run_kwargs['task_id']))
            result = json.loads(actual._invoke_tool('write_file', {'path':'draft.txt','content':run_kwargs['user_message']}, run_kwargs['task_id']))
            assert result['success']
            intervals.append((began,time.monotonic()))
            return {'final_response':'done'}
        agent.run_conversation.side_effect = run
        return agent
    app = web.Application()
    app.router.add_post('/v1/runs',adapter._handle_runs)
    app.router.add_get('/v1/runs/{run_id}',adapter._handle_get_run)
    with patch.object(adapter,'_create_agent',side_effect=create):
        async with TestClient(TestServer(app)) as client:
            requests = []
            for number in range(2):
                root = tmp_path / str(number); root.mkdir()
                policy = dict(POLICY,hash=POLICY_HASH,read_roots=[str(root)],write_roots=[str(root)])
                requests.append(client.post('/v1/runs',json={'input':f'job{number}','jentera_lane_policy':policy}))
            responses = await asyncio.gather(*requests)
            assert [r.status for r in responses] == [202,202]
            starts = [await r.json() for r in responses]
            assert all(s['jentera_lane_policy_applied']==POLICY_HASH for s in starts)
            await asyncio.wait_for(asyncio.gather(*list(adapter._active_run_tasks.values())),timeout=10)
            for start in starts:
                response = await client.get('/v1/runs/'+start['run_id'])
                status = await response.json()
                assert status['status']=='completed', status
                assert status['executionEnded'] is True
                assert status['jentera_lane_policy_applied']==POLICY_HASH
            assert len(intervals)==2
            assert max(i[0] for i in intervals)<min(i[1] for i in intervals)
            assert (tmp_path/'0'/'draft.txt').read_text()=='job0'
            assert (tmp_path/'1'/'draft.txt').read_text()=='job1'

@pytest.mark.asyncio
async def test_root_terminal_is_not_end_proof_while_tool_worker_is_alive(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    started = threading.Event(); release = threading.Event(); finished = threading.Event()
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent._tool_worker_threads = set(); agent._jentera_tool_futures = set(); agent._tool_worker_threads_lock = threading.Lock()
    agent._pending_steer = None
    def run(**kwargs):
        def worker():
            tid = threading.get_ident()
            with agent._tool_worker_threads_lock: agent._tool_worker_threads.add(tid)
            started.set()
            try:
                release.wait(timeout=5)
                (tmp_path/'late.txt').write_text('late effect')
            finally:
                with agent._tool_worker_threads_lock: agent._tool_worker_threads.discard(tid)
                finished.set()
        threading.Thread(target=worker).start()
        assert started.wait(timeout=2)
        return {'final_response':'root returned before tool'}
    agent.run_conversation.side_effect = run
    app = web.Application(); app.router.add_post('/v1/runs',adapter._handle_runs); app.router.add_get('/v1/runs/{run_id}',adapter._handle_get_run)
    try:
        with patch.object(adapter,'_create_agent',return_value=agent):
            async with TestClient(TestServer(app)) as client:
                response = await client.post('/v1/runs',json={'input':'synthetic'})
                run_id = (await response.json())['run_id']
                await asyncio.wait_for(asyncio.gather(*list(adapter._active_run_tasks.values())),timeout=5)
                assert run_id not in adapter._active_run_agents
                status = await (await client.get('/v1/runs/'+run_id)).json()
                assert status['status']=='completed'
                assert status['executionEnded'] is False
                release.set(); assert await asyncio.to_thread(finished.wait,2)
                status = await (await client.get('/v1/runs/'+run_id)).json()
                assert status['executionEnded'] is True
    finally:
        release.set()

@pytest.mark.asyncio
async def test_constructor_error_is_known_never_executed_after_executor_ends():
    adapter = APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    app=web.Application(); app.router.add_post('/v1/runs',adapter._handle_runs); app.router.add_get('/v1/runs/{run_id}',adapter._handle_get_run)
    with patch.object(adapter,'_create_agent',side_effect=RuntimeError('synthetic constructor error')):
        async with TestClient(TestServer(app)) as client:
            response=await client.post('/v1/runs',json={'input':'synthetic'})
            run_id=(await response.json())['run_id']
            await asyncio.wait_for(asyncio.gather(*list(adapter._active_run_tasks.values())),timeout=5)
            status=await (await client.get('/v1/runs/'+run_id)).json()
            assert status['status']=='failed'
            assert status['executionEnded'] is True

@pytest.mark.asyncio
async def test_running_future_before_worker_registration_retains_end_proof():
    from concurrent.futures import ThreadPoolExecutor
    from agent.jentera_lane_policy import track_tool_future
    adapter=APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    entered=threading.Event(); release=threading.Event()
    agent=MagicMock(); agent.session_prompt_tokens=agent.session_completion_tokens=agent.session_total_tokens=0
    agent._tool_worker_threads=set(); agent._tool_worker_threads_lock=threading.Lock(); agent._pending_steer=None
    executor=ThreadPoolExecutor(max_workers=1)
    def before_registration():
        entered.set(); release.wait(timeout=5)
    def run(**kwargs):
        future=executor.submit(before_registration)
        track_tool_future(agent,future)
        assert entered.wait(timeout=2)
        return {'final_response':'root finished, submitted tool not registered yet'}
    agent.run_conversation.side_effect=run
    app=web.Application(); app.router.add_post('/v1/runs',adapter._handle_runs); app.router.add_get('/v1/runs/{run_id}',adapter._handle_get_run)
    try:
        with patch.object(adapter,'_create_agent',return_value=agent):
            async with TestClient(TestServer(app)) as client:
                response=await client.post('/v1/runs',json={'input':'synthetic'})
                run_id=(await response.json())['run_id']
                await asyncio.wait_for(asyncio.gather(*list(adapter._active_run_tasks.values())),timeout=5)
                assert not agent._tool_worker_threads
                assert agent._jentera_tool_futures
                status=await (await client.get('/v1/runs/'+run_id)).json()
                assert status['executionEnded'] is False
                release.set(); await asyncio.to_thread(executor.shutdown,wait=True)
                status=await (await client.get('/v1/runs/'+run_id)).json()
                assert status['executionEnded'] is True
    finally:
        release.set(); executor.shutdown(wait=True)

def test_status_ttl_retains_live_owned_workers_then_keeps_lightweight_end_proof():
    from types import SimpleNamespace
    from concurrent.futures import Future
    adapter=APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    future=Future(); future.set_running_or_notify_cancel()
    agent=SimpleNamespace(_tool_worker_threads_lock=threading.Lock(),_tool_worker_threads=set(),_jentera_tool_futures={future})
    adapter._run_statuses['owned']={'run_id':'owned','status':'completed','updated_at':1,'output':'ACTUAL_RESULT','usage':{'input_tokens':12,'output_tokens':7},'pending_steer':'Owner correction','error':'bounded error','last_event':'run.completed','diagnostic_blob':'discard me'}
    adapter._jentera_ended_threads.add('owned'); adapter._jentera_proof_agents['owned']=agent
    adapter._sweep_orphaned_runs_once(7200)
    assert 'owned' in adapter._jentera_proof_agents
    assert not adapter._jentera_execution_ended('owned',adapter._run_statuses['owned'])
    future.set_result(None); agent._jentera_tool_futures.clear()
    adapter._sweep_orphaned_runs_once(7201)
    assert adapter._run_statuses['owned']['executionEnded'] is True
    for key in ['output','usage','pending_steer','error','last_event']:
        expected={'output':'ACTUAL_RESULT','usage':{'input_tokens':12,'output_tokens':7},'pending_steer':'Owner correction','error':'bounded error','last_event':'run.completed'}
        assert adapter._run_statuses['owned'][key] == expected[key]
    assert 'diagnostic_blob' not in adapter._run_statuses['owned']
    assert 'owned' not in adapter._jentera_proof_agents
    adapter._sweep_orphaned_runs_once(7*24*3600)
    assert adapter._jentera_execution_ended('owned',adapter._run_statuses['owned'])
    assert adapter._run_statuses['owned']['output']=='ACTUAL_RESULT'
    assert adapter._run_statuses['owned']['pending_steer']=='Owner correction'
