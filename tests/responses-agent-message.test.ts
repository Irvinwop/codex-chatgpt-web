import { expect, test } from "bun:test";
import { compileChatGptWebPrompt } from "../src/adapters/chatgpt-web/prompt";
import { CHATGPT_WEB_MODEL_ID } from "../src/adapters/chatgpt-web/model";
import { parseRequest } from "../src/responses/parser";

const capabilities = { localToolsEnabled: true, solAvailable: true, proAvailable: true };
const turnToken = "turn_12345678901234567890123456789012";

function request() {
  return parseRequest({
    model: CHATGPT_WEB_MODEL_ID,
    stream: true,
    instructions: "Preserve the native conversation semantics.",
    input: [
      {
        type: "agent_message",
        author: "parent",
        recipient: "child",
        content: [{ type: "input_text", text: "Inspect the failing request and report evidence." }],
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Continue from the agent report." }],
      },
    ],
  });
}

function inlineMessages(text: string): Array<Record<string, unknown>> {
  const match = text.match(/<codex_context_json>\n([^\n]+)\n<\/codex_context_json>/);
  if (!match?.[1]) throw new Error("inline Codex context JSON missing");
  return (JSON.parse(match[1]) as { messages: Array<Record<string, unknown>> }).messages;
}

test("parser preserves plaintext agent-message routing metadata", () => {
  const parsed = request();
  expect(parsed.context.messages[0]).toEqual({
    role: "agentMessage",
    author: "parent",
    recipient: "child",
    content: "Inspect the failing request and report evidence.",
    timestamp: expect.any(Number),
  });
});

test("inline Web context emits a distinct agent_message envelope", () => {
  const compiled = compileChatGptWebPrompt(request(), capabilities, turnToken);
  const messages = inlineMessages(compiled.text);
  expect(messages[0]).toEqual({
    role: "agent_message",
    author: "parent",
    recipient: "child",
    content: "Inspect the failing request and report evidence.",
  });
  expect(messages[1]).toEqual({
    role: "user",
    content: "Continue from the agent report.",
  });
  expect(compiled.text).toContain("agent_message messages are inter-agent inputs");
  expect(compiled.text).toContain("Exclude agent_message inputs");
});

test("multipart Web context emits the same agent_message envelope", () => {
  const compiled = compileChatGptWebPrompt(
    request(),
    capabilities,
    turnToken,
    { experimentalMultipartParts: 2 },
  );
  const records = compiled.multipart!.parts.flatMap(part => (
    (JSON.parse(part) as { records: Array<Record<string, unknown>> }).records
  ));
  const messages = records
    .filter(record => record.kind === "message")
    .map(record => record.message as Record<string, unknown>);
  expect(messages[0]).toEqual({
    role: "agent_message",
    author: "parent",
    recipient: "child",
    content: "Inspect the failing request and report evidence.",
  });
});

test("ordinary user messages do not gain agent metadata", () => {
  const messages = inlineMessages(compileChatGptWebPrompt(request(), capabilities, turnToken).text);
  expect(messages[1]).not.toHaveProperty("author");
  expect(messages[1]).not.toHaveProperty("recipient");
});

test("agent messages do not invent missing routing identity or fallback content", () => {
  const parsed = parseRequest({
    model: CHATGPT_WEB_MODEL_ID,
    stream: true,
    input: [{ type: "agent_message", content: "" }],
  });
  expect(parsed.context.messages[0]).toMatchObject({ role: "agentMessage", content: "" });
  expect(parsed.context.messages[0]).not.toHaveProperty("author");
  expect(parsed.context.messages[0]).not.toHaveProperty("recipient");
  const messages = inlineMessages(compileChatGptWebPrompt(parsed, capabilities, turnToken).text);
  expect(messages[0]).toEqual({ role: "agent_message", content: "" });
});

test("historical encrypted V2 agent messages retain routing and degrade only the opaque body", () => {
  const parsed = parseRequest({
    model: CHATGPT_WEB_MODEL_ID,
    stream: true,
    client_metadata: {
      "x-codex-turn-metadata": JSON.stringify({
        thread_id: "thread_history",
        turn_id: "turn_current",
      }),
    },
    input: [
      {
        type: "agent_message",
        author: "/root/worker",
        recipient: "/root",
        content: [
          { type: "input_text", text: "Message Type: FINAL_ANSWER\nPayload:\n" },
          { type: "encrypted_content", encrypted_content: "gAAAAABhistorical-native-v2-payload" },
        ],
        internal_chat_message_metadata_passthrough: { turn_id: "turn_historical" },
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Continue with the preserved task history." }],
        internal_chat_message_metadata_passthrough: { turn_id: "turn_current" },
      },
    ],
  });

  expect(parsed._opaqueMultiAgentV2Payload).toBeUndefined();
  const historicalMessage = parsed.context.messages[0] as { content: string };
  expect(historicalMessage).toMatchObject({
    role: "agentMessage",
    author: "/root/worker",
    recipient: "/root",
  });
  expect(typeof historicalMessage.content).toBe("string");
  expect(historicalMessage.content.includes("Historical encrypted cross-backend subagent payload omitted")).toBe(true);
  expect(historicalMessage.content.includes("Message Type: FINAL_ANSWER")).toBe(true);
});

test("current-turn encrypted V2 agent messages still fail closed", () => {
  const parsed = parseRequest({
    model: CHATGPT_WEB_MODEL_ID,
    stream: true,
    client_metadata: {
      "x-codex-turn-metadata": JSON.stringify({
        thread_id: "thread_child",
        turn_id: "turn_child",
      }),
    },
    input: [{
      type: "agent_message",
      author: "/root",
      recipient: "/root/child",
      content: [
        { type: "input_text", text: "Message Type: NEW_TASK\nPayload:\n" },
        { type: "encrypted_content", encrypted_content: "gAAAAABcurrent-native-v2-payload" },
      ],
      internal_chat_message_metadata_passthrough: { turn_id: "turn_child" },
    }],
  });

  expect(parsed._opaqueMultiAgentV2Payload).toBe(true);
});
