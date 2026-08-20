import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.errors import rate_limit_message
from agent.graph import compact_history, is_full_squad_request


class CompactHistoryTests(unittest.TestCase):
    def test_removes_completed_tool_traces(self):
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="Pick my team"),
            AIMessage(content="", tool_calls=[{
                "name": "build_squad",
                "args": {"save_result": True},
                "id": "call-1",
                "type": "tool_call",
            }]),
            ToolMessage(content="large squad result", tool_call_id="call-1"),
            AIMessage(content="Here is your squad"),
        ]

        compacted = compact_history(messages)

        self.assertEqual(
            [message.content for message in compacted],
            ["Pick my team", "Here is your squad"],
        )

    def test_keeps_only_the_requested_number_of_recent_turns(self):
        messages = []
        for number in range(6):
            messages.extend([
                HumanMessage(content=f"question {number}"),
                AIMessage(content=f"answer {number}"),
            ])

        compacted = compact_history(messages, max_turns=2)

        self.assertEqual(
            [message.content for message in compacted],
            ["question 4", "answer 4", "question 5", "answer 5"],
        )


class SquadRoutingTests(unittest.TestCase):
    def test_routes_explicit_full_team_request(self):
        self.assertTrue(is_full_squad_request([
            HumanMessage(content="Pick my team for this gameweek"),
        ]))

    def test_does_not_route_single_player_question(self):
        self.assertFalse(is_full_squad_request([
            HumanMessage(content="Who should I pick for my midfield?"),
        ]))


class RateLimitMessageTests(unittest.TestCase):
    def test_identifies_token_limit_and_retry_time(self):
        message = rate_limit_message(
            Exception("429 tokens per minute exceeded; try again in 12.5s")
        )

        self.assertIn("token-per-minute", message)
        self.assertIn("12.5 s", message)


if __name__ == "__main__":
    unittest.main()
