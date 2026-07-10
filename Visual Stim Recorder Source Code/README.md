# VisualStimRecorder2 - Vuzix AR Visual Stimulation Recorder

A specialized Android application designed for Vuzix AR glasses that displays visual stimuli (red blocks) and records timing events via Lab Streaming Layer (LSL) integration.

## Overview

This application is specifically designed for research and experimental use with Vuzix AR devices. It presents visual stimuli at randomized intervals while streaming precise timing markers through LSL for synchronization with other research equipment and software.

## Features

- **Visual Stimulation**: Displays red square blocks (80% of screen height) on a black background
- **Randomized Timing**: Uses Gaussian distribution for stimulus intervals (1-5 seconds) and durations (75-225ms)
- **LSL Integration**: Streams precise timing markers for each stimulus presentation
- **WiFi Monitoring**: Displays warnings when WiFi connectivity is lost
- **Wake Lock Management**: Keeps the device screen active during operation
- **Fullscreen Mode**: Optimized for AR glasses with immersive fullscreen display
- **Touch Control**: Simple tap-to-toggle interface for starting/stopping stimulation

## System Requirements

- **Android Device**: Vuzix AR glasses or Android device (API level 30+)
- **WiFi Network**: Stable WiFi connection for LSL streaming
- **LSL-Compatible Software**: Compatible LSL receiver application for marker collection

## Installation

### Building from Source

1. **Prerequisites**:
   - Android Studio Arctic Fox or newer
   - Android SDK with API level 30+
   - CMake 3.22.1+
   - Android NDK

2. **Clone and Build**:
   ```bash
   git clone <repository-url>
   cd VisualStimRecorder2
   ./gradlew assembleDebug
   ```

3. **Install**:
   - Connect your Vuzix device via USB
   - Enable Developer Options and USB Debugging
   - Run: `./gradlew installDebug`

### Sideload Installation

1. Build the APK as described above
2. Transfer the APK file to your Vuzix device
3. Use a file manager to install the APK
4. Grant necessary permissions (WiFi, Wake Lock)

## Usage

### Basic Operation

1. **Launch the Application**:
   - The app starts in fullscreen mode with a black screen
   - Tap anywhere to start the visual stimulation

2. **Starting Stimulation**:
   - Tap the screen to begin showing red blocks
   - Text "Tap to resume" appears when stopped

3. **During Operation**:
   - Red blocks appear at random intervals (1-5 seconds apart)
   - Each block is visible for 75-225 milliseconds
   - LSL markers are sent for each block presentation
   - WiFi connection is continuously monitored

4. **Stopping Stimulation**:
   - Tap the screen again to stop the stimulation
   - LSL streaming stops automatically
   - Wake lock is released

### LSL Integration

The application automatically:
- Initializes LSL connection when stimulation starts
- Sends a marker for each visual block presentation
- Stops LSL streaming when stimulation ends
- Handles connection loss gracefully

**LSL Stream Details**:
- Stream Name: `VisualStimRecorder`
- Stream Type: `Markers`
- Channel Count: 1
- Sampling Rate: Event-driven (irregular)

### WiFi Requirements

- **Stable Connection**: Required for LSL functionality
- **Network Monitoring**: App displays "Wi-Fi weak" warning when connection is lost
- **Auto-recovery**: Automatically resumes LSL streaming when connection is restored

## Technical Details

### Architecture

- **Native Code**: C++ integration via JNI for LSL functionality
- **UI Framework**: Jetpack Compose for modern Android UI
- **Concurrency**: Kotlin Coroutines for timing operations
- **Power Management**: Android WakeLock API for screen control

### Key Components

- `MainActivity`: Main application lifecycle and native interface
- `VisualStimApp`: Core stimulation logic and UI
- `WifiWarning`: Network connectivity monitoring
- `Utils`: Helper functions and Gaussian random generation
- `native-lib.cpp`: C++ LSL integration code

### Dependencies

- **AndroidX**: Core Android libraries
- **Jetpack Compose**: Modern UI toolkit
- **Kotlinx Coroutines**: Asynchronous programming
- **LSL Library**: Lab Streaming Layer (included as prebuilt binaries)

## Troubleshooting

### Common Issues

**LSL Connection Fails**:
- Ensure WiFi is connected and stable
- Check firewall settings on receiving computer
- Verify LSL library is properly loaded

**Screen Doesn't Stay Awake**:
- Check that the app has wake lock permission
- Ensure battery optimization is disabled for the app

**Visual Blocks Don't Appear**:
- Verify the app is in fullscreen mode
- Check that the device screen is functioning properly

**Performance Issues**:
- Close other applications to free up resources
- Ensure device has sufficient battery

### Debug Mode

Enable USB debugging and monitor Android Studio logs:
```bash
adb logcat | grep VisualStimRecorder
```

## Development

### Project Structure
```
app/src/main/
├── java/com/agili8/visualstimrecorder/
│   ├── MainActivity.kt      # Main application activity
│   ├── VisualStimApp.kt     # Core stimulation composable
│   ├── WifiWarning.kt       # Network monitoring UI
│   └── Utils.kt            # Utility functions
├── cpp/
│   ├── CMakeLists.txt      # Native build configuration
│   └── native-lib.cpp      # LSL integration code
└── jniLibs/                # Prebuilt LSL libraries
```

### Building Native Code

The project includes prebuilt LSL libraries for common architectures:
- `armeabi-v7a` (ARM 32-bit)
- `arm64-v8a` (ARM 64-bit)
- `x86` (Intel 32-bit)
- `x86_64` (Intel 64-bit)

For custom builds, modify `CMakeLists.txt` and rebuild:
```bash
./gradlew assembleDebug
```

## License

This project is developed for research purposes. Please ensure compliance with your institution's research ethics requirements when using this software.

## Support

For technical support or feature requests, please contact the development team or submit an issue in the project repository.

---

*Designed for Vuzix AR devices and research applications requiring precise visual stimulation timing and synchronization.*
