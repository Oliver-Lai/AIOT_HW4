# Specification: Character Recognition

## ADDED Requirements

### Requirement: Image Preprocessing for Inference Must Match Training Format
User-drawn images from the canvas SHALL be preprocessed to match the EMNIST training format before model inference.

**Properties**:
- Input format: RGBA image from drawing canvas (variable size, typically 280x280 or larger)
- Output format: Grayscale image (28, 28, 1) normalized to [0, 1]
- Transformations: Resize, color inversion, normalization, reshaping
- Processing time: <100ms per image

#### Scenario: Preprocess canvas drawing for prediction
**Given** a user has drawn a character on the canvas  
**And** the canvas image is captured as RGBA format  
**When** the preprocessing function is called  
**Then** the alpha channel is extracted as the drawing data  
**And** the image is resized to 28x28 pixels using area interpolation  
**And** colors are inverted (black-on-white becomes white-on-black)  
**And** pixel values are normalized to range [0, 1]  
**And** the image is reshaped to (1, 28, 28, 1) for batch inference

#### Scenario: Handle empty or invalid canvas
**Given** a blank canvas with no drawings  
**When** preprocessing is attempted  
**Then** the system detects the canvas is empty (all pixels near 0)  
**And** an error message is returned indicating no content to predict  
**And** the prediction process is halted without model inference

#### Scenario: Validate preprocessing output
**Given** a preprocessed image ready for inference  
**When** validation checks run  
**Then** the image shape is exactly (1, 28, 28, 1)  
**And** all pixel values are within [0, 1] range  
**And** the image dtype is float32  
**And** the image contains at least 5% non-zero pixels

### Requirement: Character Prediction Must Use Trained Model
The system SHALL use the trained model to predict characters from preprocessed images.

**Properties**:
- Input: Preprocessed image tensor (1, 28, 28, 1)
- Output: Probability distribution over 62 character classes
- Inference mode: Single image prediction
- Response time: <2 seconds total (preprocessing + inference + post-processing)

#### Scenario: Predict character from user drawing
**Given** a preprocessed image from the canvas  
**And** the trained model is loaded in memory  
**When** the prediction function is called  
**Then** the model performs inference on the image  
**And** a probability vector of length 62 is returned  
**And** probabilities sum to approximately 1.0  
**And** inference completes in under 1 second

#### Scenario: Handle low confidence predictions
**Given** a model prediction with top probability below 50%  
**When** the prediction result is processed  
**Then** a warning flag is set indicating low confidence  
**And** the warning is displayed to the user  
**And** predictions are still shown but marked as uncertain  
**And** the user is prompted to redraw more clearly

### Requirement: Top-K Prediction Ranking Must Display Confidence Scores
The system SHALL rank and display the top most likely character predictions with confidence scores.

**Properties**:
- K value: Top 5 predictions displayed
- Confidence format: Percentage with 1 decimal place
- Sorting: Descending order by probability
- Character mapping: Indices converted to readable characters using label_mapping.json

#### Scenario: Display top 5 predictions
**Given** a probability vector from model inference  
**When** top-K ranking is performed  
**Then** the 5 highest probability classes are identified  
**And** class indices are mapped to characters (digits/letters)  
**And** probabilities are converted to percentages  
**And** results are formatted as "Character (Confidence%)"  
**And** results are sorted from highest to lowest confidence

#### Scenario: Map class indices to characters
**Given** predicted class indices [15, 23, 8, 42, 55]  
**And** the label mapping file is loaded  
**When** character mapping is performed  
**Then** each index is converted to its corresponding character  
**And** digits (0-9) map to indices 0-9  
**And** uppercase letters (A-Z) map to indices 10-35  
**And** lowercase letters (a-z) map to indices 36-61  
**And** mapping handles all 62 classes correctly

### Requirement: Real-Time Inference Must Provide Responsive Predictions
The character recognition system SHALL provide responsive predictions for interactive use.

**Properties**:
- Total latency: <2 seconds from button click to result display
- Model loading: Lazy load on first use or pre-load at startup
- Caching: Model remains in memory between predictions
- Error recovery: Graceful handling of inference failures

#### Scenario: First prediction with cold model
**Given** the Streamlit app has just started  
**And** the model has not been loaded yet  
**When** the user draws a character and clicks "Predict"  
**Then** the model is loaded from disk  
**And** the loading process takes 2-5 seconds  
**And** a loading spinner is displayed during model loading  
**And** prediction proceeds once model is ready  
**And** total time to first prediction is under 7 seconds

#### Scenario: Subsequent predictions with warm model
**Given** the model is already loaded in memory  
**And** the user draws a new character  
**When** the user clicks "Predict"  
**Then** preprocessing begins immediately  
**And** inference completes in under 1 second  
**And** results are displayed within 2 seconds total  
**And** no model reloading occurs

#### Scenario: Handle inference errors
**Given** the model is loaded  
**When** an unexpected error occurs during inference  
**Then** the error is caught and logged  
**And** a user-friendly error message is displayed  
**And** the application remains functional  
**And** the user can try drawing again without restart

### Requirement: Prediction History Must Display Recent Predictions
The system SHALL optionally display recent predictions for user reference.

**Properties**:
- History size: Last 5 predictions
- Display format: Chronological list with timestamp
- Storage: Session-based (cleared on page refresh)
- Optional feature: Can be toggled on/off in UI

#### Scenario: Track prediction history
**Given** the user has made multiple predictions  
**When** a new prediction is made  
**Then** the prediction result is added to history  
**And** history displays the last 5 predictions  
**And** each entry shows the top predicted character and confidence  
**And** oldest predictions are removed when history exceeds 5 entries

#### Scenario: Clear prediction history
**Given** the prediction history contains entries  
**When** the user clears the canvas or resets the app  
**Then** all prediction history is cleared  
**And** the history display is empty  
**And** memory used by history is released

### Requirement: Character Class Support Must Include All EMNIST Classes
The recognition system SHALL support all 62 character classes from EMNIST ByClass.

**Properties**:
- Digits: 0-9 (10 classes)
- Uppercase letters: A-Z (26 classes)
- Lowercase letters: a-z (26 classes)
- Total: 62 distinct character classes
- Case sensitivity: Model distinguishes between upper and lowercase

#### Scenario: Recognize digit characters
**Given** a user draws a digit (0-9)  
**When** prediction is performed  
**Then** the model identifies the digit correctly  
**And** the top prediction is a digit character  
**And** confidence score is displayed  
**And** other digit candidates may appear in top-5 if similar

#### Scenario: Recognize uppercase letters
**Given** a user draws an uppercase letter (A-Z)  
**When** prediction is performed  
**Then** the model identifies the uppercase letter  
**And** the top prediction is an uppercase character  
**And** similar-looking lowercase letters may rank lower  
**And** confusable letters (e.g., O vs 0) are differentiated

#### Scenario: Recognize lowercase letters
**Given** a user draws a lowercase letter (a-z)  
**When** prediction is performed  
**Then** the model identifies the lowercase letter  
**And** the top prediction is a lowercase character  
**And** case-sensitive distinction from uppercase is maintained  
**And** letters with similar shapes (e.g., c, C, o, O) are handled

#### Scenario: Handle ambiguous characters
**Given** a user draws a character that could be multiple classes (e.g., O, o, 0)  
**When** prediction is performed  
**Then** the model returns probabilities for all similar classes  
**And** the most likely class is shown first  
**And** alternative interpretations appear in top-5 predictions  
**And** the user can see why the prediction might be ambiguous
