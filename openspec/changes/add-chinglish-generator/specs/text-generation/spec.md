# Text Generation Capability

## ADDED Requirements

### Requirement: Local LLM Integration
The system SHALL integrate with pre-downloaded local language models using Hugging Face transformers to enable text generation without runtime downloads or external API dependencies.

#### Scenario: Successfully load model from local directory
- **GIVEN** model files exist in models/ directory
- **WHEN** application starts
- **THEN** model is loaded from local files using transformers library
- **AND** model is cached for subsequent use
- **AND** no network requests are made

#### Scenario: Handle missing model files
- **GIVEN** model files do not exist in models/ directory
- **WHEN** application attempts to load model
- **THEN** application displays clear error message
- **AND** provides instructions to run model download script
- **AND** does not attempt to download model automatically

### Requirement: Chinglish Style Text Generation
The system SHALL generate text in Chinglish style (晶晶體) that naturally mixes Chinese and English based on user-provided topics.

#### Scenario: Generate Chinglish text from simple topic
- **GIVEN** user provides topic "上課遲到" (late to class)
- **WHEN** generation is requested
- **THEN** system returns 100-150 character mixed language text
- **AND** text includes natural code-switching between Chinese and English
- **AND** common English terms are used appropriately (e.g., class, professor, deadline)

#### Scenario: Generate text with technical topic
- **GIVEN** user provides topic "programming homework"
- **WHEN** generation is requested
- **THEN** generated text uses technical English terms naturally
- **AND** conversational Chinese phrases connect the ideas
- **AND** style mimics overseas student speaking patterns

#### Scenario: Handle empty or invalid topic
- **GIVEN** user provides empty string or whitespace only
- **WHEN** generation is requested
- **THEN** system returns validation error
- **AND** prompts user to provide valid topic

### Requirement: Prompt Engineering and Templates
The system SHALL use engineered prompts to guide the LLM toward authentic Chinglish style output.

#### Scenario: Apply system prompt for style consistency
- **GIVEN** any user topic
- **WHEN** sending request to LLM
- **THEN** system prepends curated system prompt
- **AND** prompt includes Chinglish speaking patterns and examples
- **AND** instructs model to act as overseas student

#### Scenario: Format user topic in prompt template
- **GIVEN** user topic "週末計畫"
- **WHEN** constructing LLM prompt
- **THEN** topic is inserted into predefined template
- **AND** template includes context and style guidelines
- **AND** specifies desired output length

### Requirement: Text Generation Performance
The system SHALL complete text generation within acceptable time limits to maintain good user experience.

#### Scenario: Generate text within timeout limit
- **GIVEN** valid user topic
- **WHEN** generation is initiated
- **THEN** response is returned within 30 seconds
- **AND** generation time is logged for monitoring

#### Scenario: Handle generation timeout
- **GIVEN** LLM takes longer than 30 seconds
- **WHEN** timeout occurs
- **THEN** request is cancelled
- **AND** user receives timeout error message
- **AND** user can retry with same or different topic

### Requirement: Response Validation
The system SHALL validate LLM output to ensure quality and safety before returning to user.

#### Scenario: Validate output contains both languages
- **GIVEN** LLM returns generated text
- **WHEN** validating response
- **THEN** system checks for presence of both Chinese and English characters
- **AND** rejects output if entirely single-language
- **AND** may retry generation once if validation fails

#### Scenario: Filter inappropriate content
- **GIVEN** LLM returns text with inappropriate content
- **WHEN** validating response
- **THEN** system detects problematic content
- **AND** returns generic error without displaying content
- **AND** logs incident for review

### Requirement: Model Configuration
The system SHALL support configuration of LLM model parameters for flexibility and optimization.

#### Scenario: Use configured default model
- **GIVEN** system configuration specifies Qwen2-1.5B-Instruct
- **WHEN** application initializes
- **THEN** uses specified model from models/ directory for all generations
- **AND** logs model name and version

#### Scenario: Allow model switching via configuration
- **GIVEN** administrator changes model path in config file
- **WHEN** application restarts
- **THEN** new model is loaded from specified directory
- **AND** generation continues with new model
- **AND** no code changes required

### Requirement: Error Recovery
The system SHALL handle LLM errors gracefully and provide useful feedback to users.

#### Scenario: Retry on transient LLM error
- **GIVEN** LLM returns temporary error (e.g., memory allocation failure, timeout)
- **WHEN** error is detected
- **THEN** system retries request once after brief delay
- **AND** returns result if retry succeeds
- **AND** returns error if retry also fails

#### Scenario: Log errors for debugging
- **GIVEN** any LLM error occurs
- **WHEN** handling the error
- **THEN** full error details are logged
- **AND** log includes timestamp, topic, model used, and error message
- **AND** user receives sanitized error message
