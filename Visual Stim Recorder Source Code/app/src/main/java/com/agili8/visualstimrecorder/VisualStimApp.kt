package com.agili8.visualstimrecorder

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.random.Random
import kotlinx.coroutines.delay
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.DurationUnit

@Composable
fun VisualStimApp(
    onBlockShown: () -> Unit,
    onRunningChanged: (Boolean) -> Unit
) {
    var running by remember { mutableStateOf(false) }
    var showBlock by remember { mutableStateOf(false) }
    var totalRunningTime by remember { mutableStateOf(0L) }
    var sessionStartTime by remember { mutableStateOf(0L) }

    LaunchedEffect(running) {
        onRunningChanged(running)
        if (running) {
            sessionStartTime = System.currentTimeMillis()
        } else {
            // Add elapsed time to total when stopping
            if (sessionStartTime > 0) {
                totalRunningTime += System.currentTimeMillis() - sessionStartTime
                sessionStartTime = 0
            }
        }
        while (running) {
            val interval = (Random.nextGaussian() * 1000 + 3000).toLong().coerceIn(1000, 5000)
            delay(interval)

            val visibleDuration = (Random.nextGaussian() * 10 + 125).toLong().coerceIn(75, 225)
            showBlock = true
            onBlockShown()
            delay(visibleDuration)
            showBlock = false
        }
    }

    fun formatDuration(millis: Long): String {
        val duration = millis.milliseconds
        val hours = duration.inWholeHours
        val minutes = duration.inWholeMinutes % 60
        val seconds = duration.inWholeSeconds % 60

        return when {
            hours > 0 -> String.format("%02d:%02d:%02d", hours, minutes, seconds)
            minutes > 0 -> String.format("%02d:%02d", minutes, seconds)
            else -> String.format("00:%02d", seconds)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .clickable { running = !running },
        contentAlignment = Alignment.Center
    ) {
        when {
            !running -> Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "Tap to resume",
                    color = Color.White,
                    fontSize = 32.sp,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Total time: ${formatDuration(totalRunningTime)}",
                    color = Color.LightGray,
                    fontSize = 24.sp,
                    textAlign = TextAlign.Center
                )
            }
            showBlock -> Box(
                modifier = Modifier
                    .fillMaxHeight(0.80f)  // 85% of screen height
                    .aspectRatio(1f)       // make width = height
                    .background(Color.Red)
            )
        }
    }
}
