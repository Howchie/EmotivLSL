package com.agili8.visualstimrecorder

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

@Composable
fun WifiWarning(
    context: Context,
    onConnectionRecovered: () -> Unit = {},
    onConnectionLost: () -> Unit = {}
) {
    var isConnected by remember { mutableStateOf(true) }
    var wasDisconnected by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        while (true) {
            val currentConnected = checkWifi(context)

            if (!isConnected && currentConnected && wasDisconnected) {
                onConnectionRecovered()
            }
            if (isConnected && !currentConnected) {
                onConnectionLost()
            }

            if (!currentConnected && isConnected) wasDisconnected = true
            else if (currentConnected && !isConnected) wasDisconnected = false

            isConnected = currentConnected
            delay(2000)
        }
    }

    if (!isConnected) {
        Box(
            modifier = Modifier
                .padding(top = 16.dp)
                .wrapContentWidth()
                .height(28.dp)
                .background(Color(0xAA880000), shape = androidx.compose.foundation.shape.RoundedCornerShape(14.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "Wi-Fi weak",
                color = Color.White,
                fontSize = 14.sp,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }
    }
}
