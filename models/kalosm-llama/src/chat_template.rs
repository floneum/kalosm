//! Rendering of huggingface chat templates.
//!
//! The jinja setup here follows hf-chat-template (MIT OR Apache-2.0), which checks its output
//! against python transformers on real hub templates:
//! https://github.com/GregoryBolshakov/hf-chat-template

use std::borrow::Cow;
use std::fmt::Display;

use kalosm_language_model::{ChatMessage, ContentChunk, MessageType};
use minijinja::{context, Environment, ErrorKind, Value};
use minijinja_contrib::pycompat;

#[cfg(test)]
use pretty_assertions::assert_eq;

pub(crate) struct HuggingFaceChatTemplate {
    environment: Environment<'static>,
}

/// The role name a template expects. `MessageType` serializes the system prompt as "developer",
/// which huggingface templates never match on.
fn role_name(role: MessageType) -> &'static str {
    match role {
        MessageType::SystemPrompt => "system",
        MessageType::UserMessage => "user",
        MessageType::ModelAnswer => "assistant",
    }
}

/// Replace the `{% generation %}` block with an always true `{% if %}` block.
///
/// Transformers adds the tag to mark which bytes the assistant generated. It does not change the
/// output, but minijinja has no way to add a custom statement and fails to compile the template.
fn neutralize_generation_tags(source: &str) -> Cow<'_, str> {
    let mut out: Option<String> = None;
    let mut flushed = 0; // bytes of source already copied
    let mut cursor = 0;
    while let Some(found) = source[cursor..].find("{%") {
        let open = cursor + found;
        let Some(found_end) = source[open + 2..].find("%}") else {
            break; // let minijinja report the syntax error
        };
        let close = open + 2 + found_end + 2;
        let inner = &source[open + 2..close - 2];
        // keep any whitespace control marker the tag was written with
        let lead = inner.starts_with(['-', '+']);
        let trail = inner.ends_with(['-', '+']);
        let keyword = inner
            .trim_start_matches(['-', '+'])
            .trim_end_matches(['-', '+'])
            .trim();
        let replacement = match keyword {
            "generation" => "if true",
            "endgeneration" => "endif",
            _ => {
                cursor = close;
                continue;
            }
        };
        let out = out.get_or_insert_with(String::new);
        out.push_str(&source[flushed..open]);
        out.push_str("{%");
        if lead {
            out.push_str(&inner[..1]);
        }
        out.push(' ');
        out.push_str(replacement);
        out.push(' ');
        if trail {
            out.push_str(&inner[inner.len() - 1..]);
        }
        out.push_str("%}");
        flushed = close;
        cursor = close;
    }
    match out {
        Some(mut out) => {
            out.push_str(&source[flushed..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(source),
    }
}

impl HuggingFaceChatTemplate {
    pub(crate) fn create(chat_template: impl Display) -> Result<Self, minijinja::Error> {
        let chat_template = neutralize_generation_tags(&chat_template.to_string()).into_owned();
        let mut environment = Environment::new();

        // enable python compatibility methods because most models are tested with python
        environment.set_unknown_method_callback(pycompat::unknown_method_callback);

        // transformers compiles templates with both flags on, so the whitespace around block
        // tags has to be trimmed the same way here
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);

        // add the raise_exception function from huggingface templates to the environment
        let raise_exception = |err_text: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                ErrorKind::InvalidOperation,
                format!("The template raised an exception: {err_text}"),
            ))
        };
        // add the strftime_now function from huggingface templates to the environment
        let strftime_now = |format: String| -> Result<String, minijinja::Error> {
            let now = chrono::Utc::now();
            let formatted_time = now.format(&format).to_string();
            Ok(formatted_time)
        };
        environment.add_function("raise_exception", raise_exception);
        environment.add_function("strftime_now", strftime_now);

        // compile the template expression in the environment
        environment.add_template_owned("main", chat_template)?;

        Ok(Self { environment })
    }

    pub(crate) fn format(
        &self,
        bos_token: &str,
        eos_token: &str,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, minijinja::Error> {
        let tools: Option<()> = None;
        let messages = messages
            .iter()
            .map(|message| {
                let role = role_name(message.role());
                let content = message.content();
                let content: Value = if let Some(content) = content.as_str() {
                    content.into()
                } else {
                    let chunks = content
                        .chunks()
                        .iter()
                        .map(|chunk| match chunk {
                            ContentChunk::Text(text) => {
                                context! { text }
                            }
                            ContentChunk::Media(_) => {
                                context! { image => "" }
                            }
                        })
                        .collect::<Vec<_>>();
                    chunks.into()
                };
                context! { role, content }
            })
            .collect::<Vec<_>>();
        let ctx = context! { bos_token, eos_token, messages, add_generation_prompt, tools };
        let template = self.environment.get_template("main")?;
        let result = template.render(&ctx)?;
        Ok(result)
    }
}

#[test]
fn test_qwen_chat_template() {
    let template = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            "I'd like to show off how chat templating works!".to_string(),
        ),
    ];

    let result = template
        .format("<|endoftext|>", "<|im_end|>", &inputs, false)
        .unwrap();
    assert_eq!(
        result,
        r#"<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>user
I'd like to show off how chat templating works!<|im_end|>
"#
    );
}

#[test]
fn test_qwen_vl_chat_template() {
    use kalosm_language_model::{MediaChunk, MediaSource};

    let template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let template = HuggingFaceChatTemplate::create(template).unwrap();
    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            "I'd like to show off how chat templating works!".to_string(),
        ),
    ];
    let result = template
        .format("<|begin_of_text|>", "<|end_of_text|>", &inputs, false)
        .unwrap();
    assert_eq!(
        result,
        r#"<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>user
I'd like to show off how chat templating works!<|im_end|>
"#
    );

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            (
                "I'd like to show off how chat templating works!".to_string(),
                MediaChunk::new(
                    MediaSource::url("https://example.com/image.png"),
                    kalosm_language_model::MediaType::Image,
                ),
            ),
        ),
    ];
    let result = template
        .format("<|begin_of_text|>", "<|end_of_text|>", &inputs, false)
        .unwrap();
    assert_eq!(
        result,
        r#"<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>user
I'd like to show off how chat templating works!<|vision_start|><|image_pad|><|vision_end|><|im_end|>
"#
    );
}

#[test]
fn test_llama_chat_template() {
    let template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            "I'd like to show off how chat templating works!".to_string(),
        ),
    ];

    let result = template
        .format("<|begin_of_text|>", "<|end_of_text|>", &inputs, false)
        .unwrap();

    assert_eq!(
        result,
        r#"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm doing great. How can I help you today?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'd like to show off how chat templating works!<|eot_id|>"#
    )
}

#[test]
fn test_mistral_chat_template() {
    let template = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            "I'd like to show off how chat templating works!".to_string(),
        ),
    ];

    let result = template.format("<s>", "</s>", &inputs, false).unwrap();
    assert_eq!(
        result,
        r#"<s> [INST] Hello, how are you? [/INST] I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"#
    )
}

#[test]
fn test_mistral_small_chat_template() {
    let template = "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is \" + today + \".\\n\\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\\"What are some good restaurants around me?\\\" => \\\"Where are you?\\\" or \\\"When is the next flight to Tokyo\\\" => \\\"Where do you travel from?\\\")\" %}\n\n{{- bos_token }}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n{%- for message in loop_messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello, how are you?".to_string()),
        ChatMessage::new(
            MessageType::ModelAnswer,
            "I'm doing great. How can I help you today?".to_string(),
        ),
        ChatMessage::new(
            MessageType::UserMessage,
            "I'd like to show off how chat templating works!".to_string(),
        ),
    ];

    let result = template.format("<s>", "</s>", &inputs, false).unwrap();
    println!("{result}");
    let now = chrono::Utc::now().format("%Y-%m-%d").to_string();
    assert_eq!(
        result,
        format!(
            r#"<s>[SYSTEM_PROMPT]You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
Your knowledge base was last updated on 2023-10-01. The current date is {now}.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")[/SYSTEM_PROMPT][INST]Hello, how are you?[/INST]I'm doing great. How can I help you today?</s>[INST]I'd like to show off how chat templating works![/INST]"#
        )
    )
}
#[test]
fn test_zephyr_chat_template() {
    // HuggingFaceH4/zephyr-7b-beta template, output checked against python transformers
    let template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::SystemPrompt, "You are terse.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "Hello!".to_string()),
        ChatMessage::new(MessageType::ModelAnswer, "Hi.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "What is 2+2?".to_string()),
    ];

    let result = template.format("<s>", "</s>", &inputs, true).unwrap();
    assert_eq!(
        result,
        r#"<|system|>
You are terse.</s>
<|user|>
Hello!</s>
<|assistant|>
Hi.</s>
<|user|>
What is 2+2?</s>
<|assistant|>
"#
    );
}

#[test]
fn test_phi_3_chat_template() {
    // microsoft/Phi-3-mini-4k-instruct template, output checked against python transformers
    let template = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::SystemPrompt, "You are terse.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "Hello!".to_string()),
        ChatMessage::new(MessageType::ModelAnswer, "Hi.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "What is 2+2?".to_string()),
    ];

    let result = template
        .format("<s>", "<|endoftext|>", &inputs, true)
        .unwrap();
    assert_eq!(
        result,
        r#"<|system|>
You are terse.<|end|>
<|user|>
Hello!<|end|>
<|assistant|>
Hi.<|end|>
<|user|>
What is 2+2?<|end|>
<|assistant|>
"#
    );
}

#[test]
fn test_gemma_3_chat_template() {
    // google/gemma-3-4b-it template, output checked against python transformers
    let template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::SystemPrompt, "You are terse.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "Hello!".to_string()),
        ChatMessage::new(MessageType::ModelAnswer, "Hi.".to_string()),
        ChatMessage::new(MessageType::UserMessage, "What is 2+2?".to_string()),
    ];

    let result = template.format("<bos>", "<eos>", &inputs, true).unwrap();
    assert_eq!(
        result,
        r#"<bos><start_of_turn>user
You are terse.

Hello!<end_of_turn>
<start_of_turn>model
Hi.<end_of_turn>
<start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
"#
    );
}

#[test]
fn test_generation_tag_is_ignored() {
    // some hub templates (SmolLM3) wrap the answer in a `generation` block to mark which bytes
    // the model generated. It does not change the rendered text.
    let template = "{% for message in messages %}<|{{ message['role'] }}|>{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}{% endfor %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatMessage::new(MessageType::UserMessage, "Hello!".to_string()),
        ChatMessage::new(MessageType::ModelAnswer, "Hi.".to_string()),
    ];

    let result = template.format("<s>", "</s>", &inputs, false).unwrap();
    assert_eq!(result, "<|user|>Hello!<|assistant|>Hi.");
}
