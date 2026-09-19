#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from server_utils import pending_scenarios, Scenario
from persona import build_system_prompt
import re
from twilio.rest import Client

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)

GOODBYE = re.compile(
    r"\b(bye|goodbye|take care|have a (good|great) (day|one)|"
    r"thanks for your help|that'?s all i needed)\b",
    re.IGNORECASE,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

FALLBACK = Scenario(
    name="fallback",
    patient_name="Chris Hale",
    patient_dob="1979-11-14",
    goal="Ask what the office hours are",
    opening="Hi, quick question — what time do you close today?",
    tactics=[],
    success="The agent states the office hours",
)


class GoodbyeWatcher(FrameProcessor):
    """Sits between llm and tts. Watches the assistant's text for an exit."""

    def __init__(self, max_turns: int):
        super().__init__()
        self._max_turns = max_turns
        self._turns = 0
        self._buffer = ""
        self.should_end = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._turns += 1
            logger.debug(f"turn {self._turns}: {self._buffer!r}")
            if GOODBYE.search(self._buffer):
                logger.info(f"Goodbye detected after {self._turns} turns")
                self.should_end = True
            elif self._turns >= self._max_turns:
                logger.warning(f"Hit max_turns ({self._max_turns})")
                self.should_end = True

        await self.push_frame(frame, direction)

class Hangup(FrameProcessor):
    def __init__(self, watcher: GoodbyeWatcher, call_sid: str):
        super().__init__()
        self._watcher = watcher
        self._call_sid = call_sid
        self._ending = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(f"bot finished speaking, should_end={self._watcher.should_end}")

            if self._watcher.should_end and not self._ending:
                self._ending = True
                client = Client(
                    os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
                )
                client.calls(self._call_sid).update(status="completed")
                logger.info(f"Hung up {self._call_sid}")

        await self.push_frame(frame, direction)

async def run_bot(transport: BaseTransport, handle_sigint: bool, scenario: Scenario, call_sid: str):

    llm = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma3:latest"))

    #llm = GoogleLLMService(
    #    api_key=os.getenv("GOOGLE_API_KEY"),
    #    settings=GoogleLLMService.Settings(
    #        system_instruction="You are a friendly assistant making an outbound phone call. Your responses will be read aloud, so keep them concise and conversational. Avoid special characters or formatting. Begin by politely greeting the person and explaining why you're calling.",
    #    ),
    #)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(
            voice="aura-2-asteria-en",
        ),
    )

    system_prompt = build_system_prompt(scenario)
    logger.info(f"SYSTEM PROMPT ({len(system_prompt)} chars): {system_prompt[:200]}")

    context = LLMContext(
            messages=[{"role": "system", "content": system_prompt}]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    watcher = GoodbyeWatcher(scenario.max_turns)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            user_aggregator,
            llm,  # LLM
            watcher,
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            assistant_aggregator,
            Hangup(watcher, call_sid),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the outbound conversation, waiting for the user to speak first
        logger.info("Starting outbound call conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")
        await runner.cancel()

    await runner.run()
    


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport_params = {
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    # create_transport auto-detects the telephony provider, builds the matching
    # serializer (here the TwilioFrameSerializer, using TWILIO_ACCOUNT_SID /
    # TWILIO_AUTH_TOKEN), and sets add_wav_header=False, so the bot only supplies
    # the params it cares about.
    transport = await create_transport(runner_args, transport_params)

    # Personalize the bot based on the caller's numbers.
    # The call_data is available via runner_args.call_data as a typed CallData model.
    call_data = runner_args.call_data
    logger.info(f"Call data: {call_data!r}")
    scenario = pending_scenarios.pop(call_data.call_id, FALLBACK)
    to_number = call_data.to_number if call_data else None
    from_number = call_data.from_number if call_data else None
    logger.info(f"Call metadata - To: {to_number}, From: {from_number}")

    await run_bot(transport, runner_args.handle_sigint, scenario, call_data.call_id)


if __name__ == "__main__":
    # Normally this bot runs via server.py (which initiates the Twilio call and
    # connects the Media Stream to bot()).
    from pipecat.runner.run import main

    main()
