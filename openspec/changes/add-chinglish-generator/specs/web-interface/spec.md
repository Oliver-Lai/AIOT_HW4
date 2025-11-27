# Web Interface Capability

## ADDED Requirements

### Requirement: Topic Input Interface
The system SHALL provide a Streamlit web interface for users to input topics for Chinglish text generation.

#### Scenario: Display topic input form
- **GIVEN** user accesses the Streamlit application
- **WHEN** page loads
- **THEN** st.text_input widget is displayed for topic entry
- **AND** generate button (st.button) is visible
- **AND** placeholder text guides user on input format

#### Scenario: Submit topic for generation
- **GIVEN** user enters topic "visiting a museum"
- **WHEN** user clicks generate button
- **THEN** topic is processed by application logic
- **AND** loading spinner (st.spinner) is displayed
- **AND** user cannot submit new requests during processing

#### Scenario: Validate topic input length
- **GIVEN** user enters topic longer than 200 characters
- **WHEN** user attempts to submit
- **THEN** validation error is shown
- **AND** submission is prevented
- **AND** character count is displayed

### Requirement: Generated Content Display
The system SHALL display the generated Chinglish text prominently for user review.

#### Scenario: Show generated text after successful generation
- **GIVEN** text generation completes successfully
- **WHEN** displaying results
- **THEN** text is shown using st.text_area or st.markdown
- **AND** loading spinner is hidden
- **AND** audio player widget becomes visible

#### Scenario: Format mixed-language text appropriately
- **GIVEN** generated text contains Chinese and English
- **WHEN** displaying content with Streamlit
- **THEN** text uses Streamlit's default font supporting both scripts
- **AND** font size is readable (Streamlit default styling)
- **AND** line spacing is comfortable for reading

### Requirement: Audio Playback Controls
The system SHALL provide audio player controls for users to listen to generated speech.

#### Scenario: Display audio player after generation
- **GIVEN** audio file is generated successfully
- **WHEN** displaying results
- **THEN** st.audio widget is displayed with audio file
- **AND** player has built-in play/pause controls
- **AND** player has volume controls
- **AND** player shows progress bar

#### Scenario: Play generated audio
- **GIVEN** st.audio widget is displayed with audio file
- **WHEN** user clicks play button
- **THEN** audio plays through browser
- **AND** play button changes to pause button
- **AND** progress bar updates as audio plays

#### Scenario: Handle audio playback errors
- **GIVEN** audio file fails to load or play
- **WHEN** error occurs
- **THEN** error message is displayed to user
- **AND** suggests checking audio settings or trying again
- **AND** text content remains visible

### Requirement: Loading and Progress Feedback
The system SHALL provide clear feedback during text generation and audio synthesis.

#### Scenario: Show loading state during generation
- **GIVEN** generation button is clicked
- **WHEN** processing is in progress
- **THEN** st.spinner displays with status message "Generating text..."
- **AND** user sees loading animation
- **AND** user cannot trigger new generation

#### Scenario: Update progress for long operations
- **GIVEN** generation takes more than 3 seconds
- **WHEN** waiting continues
- **THEN** st.spinner continues showing status
- **AND** user is assured process is active via animation
- **AND** Streamlit prevents multiple simultaneous executions

#### Scenario: Complete loading state
- **GIVEN** processing completes successfully
- **WHEN** exiting spinner context
- **THEN** loading spinner is hidden automatically
- **AND** results are displayed
- **AND** user can initiate new generation

### Requirement: Error Display and Handling
The system SHALL display user-friendly error messages when operations fail.

#### Scenario: Display validation error
- **GIVEN** user submits invalid input (empty or too long)
- **WHEN** validation fails
- **THEN** st.error displays message explaining the issue
- **AND** error appears above results area
- **AND** user can correct input and retry

#### Scenario: Display processing error
- **GIVEN** model loading or generation fails
- **WHEN** error occurs during processing
- **THEN** st.error displays user-friendly error message
- **AND** message explains what went wrong (e.g., "Model not found")
- **AND** provides guidance on resolution

#### Scenario: Display TTS error
- **GIVEN** text-to-speech synthesis fails
- **WHEN** TTS error is caught
- **THEN** st.warning displays indicating audio unavailable
- **AND** generated text is still shown to user
- **AND** user can read text even without audio

### Requirement: Responsive Design
The system SHALL provide responsive interface that works on different screen sizes.

#### Scenario: Display properly on desktop browsers
- **GIVEN** user accesses Streamlit app on desktop (1920x1080)
- **WHEN** page loads
- **THEN** Streamlit uses its default responsive layout
- **AND** elements are centered with appropriate padding
- **AND** all controls are easily accessible

#### Scenario: Display properly on mobile devices
- **GIVEN** user accesses Streamlit app on mobile (375x667)
- **WHEN** page loads
- **THEN** Streamlit adapts layout to narrow viewport automatically
- **AND** text input spans available width with padding
- **AND** buttons are touch-friendly (Streamlit default sizing)
- **AND** text remains readable without horizontal scrolling

### Requirement: Application Flow Integration
The system SHALL integrate text generation and audio synthesis in a seamless Streamlit workflow.

#### Scenario: Execute generation workflow
- **GIVEN** user submits valid topic
- **WHEN** generate button is clicked
- **THEN** Streamlit executes generation function
- **AND** calls model for text generation
- **AND** calls TTS for audio synthesis
- **AND** displays results in same page execution

#### Scenario: Handle state between reruns
- **GIVEN** Streamlit reruns on button click
- **WHEN** executing application logic
- **THEN** model remains cached via @st.cache_resource
- **AND** previous results are cleared for new generation
- **AND** UI state is managed properly

#### Scenario: Maintain session persistence
- **GIVEN** user generates multiple outputs
- **WHEN** using the application
- **THEN** each generation replaces previous results
- **AND** audio files are accessible during session
- **AND** page does not require manual refresh

### Requirement: User Experience Enhancements
The system SHALL provide smooth, intuitive user experience with helpful features.

#### Scenario: Enable new generation after completion
- **GIVEN** generation completes successfully
- **WHEN** results are displayed
- **THEN** user can enter new topic in input field
- **AND** generate button is active for new request
- **AND** user can immediately start new generation

#### Scenario: Provide example topics
- **GIVEN** user first visits the page
- **WHEN** viewing the interface
- **THEN** example topics are shown using st.selectbox or buttons
- **AND** selecting example populates the input field
- **AND** examples cover different topic types (校園生活, 工作, 旅行)

#### Scenario: Show generation statistics
- **GIVEN** generation completes successfully
- **WHEN** results are displayed
- **THEN** st.metric or text displays generation time
- **AND** shows synthesis time separately
- **AND** provides transparency on performance

### Requirement: Accessibility
The system SHALL follow basic accessibility guidelines for inclusive design.

#### Scenario: Keyboard navigation support
- **GIVEN** user navigates using keyboard only
- **WHEN** tabbing through Streamlit interface
- **THEN** all interactive elements (input, button) are reachable
- **AND** focus indicators are clearly visible (Streamlit default)
- **AND** Enter key in input field can trigger button via Streamlit form

#### Scenario: Screen reader compatibility
- **GIVEN** user with screen reader accesses Streamlit app
- **WHEN** navigating interface
- **THEN** Streamlit widgets have built-in ARIA labels
- **AND** button purposes are clearly announced
- **AND** status messages from st.error/st.success are accessible

#### Scenario: Sufficient color contrast
- **GIVEN** Streamlit interface uses default theme
- **WHEN** rendering page
- **THEN** Streamlit's default theme provides adequate contrast
- **AND** text and interactive elements are distinguishable
- **AND** error/success messages use color plus text
