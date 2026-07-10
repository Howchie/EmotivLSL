package com.agili8.visualstimrecorder

import android.os.Bundle
import android.os.PowerManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
class MainActivity : ComponentActivity() {

    companion object {
        init {
            System.loadLibrary("visualstimrecorder")
        }
    }

    external fun initLSL()
    external fun sendMarker()
    external fun stopLSL()

    private var lslRunning = false
    private var appRunning = false
    private var lslStoppedByConnectionLoss = false
    private var wakeLock: PowerManager.WakeLock? = null

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
            "VisualStimRecorder:WakeLock"
        )
        wakeLock?.acquire() // indefinite
    }

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        wakeLock = null
    }

    private fun stopLSLIfRunning() {
        if (lslRunning) {
            stopLSL()
            lslRunning = false
            lslStoppedByConnectionLoss = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Fullscreen
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        setContent {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.TopCenter
            ) {
                val context = LocalContext.current
                VisualStimApp(
                    onBlockShown = {
                        if (lslRunning) sendMarker()
                    },
                    onRunningChanged = { running ->
                        appRunning = running
                        if (running) {
                            if (!lslRunning) {
                                initLSL()
                                lslRunning = true
                                lslStoppedByConnectionLoss = false
                            }
                            (context as MainActivity).acquireWakeLock()
                        } else {
                            (context as MainActivity).releaseWakeLock()
                        }
                    }
                )
                WifiWarning(
                    context = context,
                    onConnectionRecovered = {
                        if (appRunning && lslStoppedByConnectionLoss && !lslRunning) {
                            initLSL()
                            lslRunning = true
                            lslStoppedByConnectionLoss = false
                            (context as MainActivity).acquireWakeLock()
                        }
                    },
                    onConnectionLost = {
                        if (appRunning && lslRunning) {
                            // stopLSL()
                            lslRunning = false
                            lslStoppedByConnectionLoss = true
                            (context as MainActivity).releaseWakeLock()
                        }
                    }
                )
            }
        }
    }

    override fun onPause() {
        super.onPause()
        // Called when activity loses focus (still partially visible)
        stopLSLIfRunning()
        releaseWakeLock()
    }

    override fun onStop() {
        super.onStop()
        stopLSLIfRunning()
        // Called when activity is no longer visible (minimized)
        // Add any cleanup for minimized state here
    }

    override fun onResume() {
        super.onResume()
        initLSL()
        // Called when activity comes back to foreground
        // Add any restoration logic here
    }

    override fun onDestroy() {
        super.onDestroy()
        stopLSLIfRunning()
        releaseWakeLock()
    }
}
