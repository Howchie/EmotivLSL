package com.agili8.visualstimrecorder

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import kotlin.random.Random

fun checkWifi(context: Context): Boolean {
    val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    val network = cm.activeNetwork ?: return false
    val capabilities = cm.getNetworkCapabilities(network) ?: return false
    return capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
}

fun Random.nextGaussian(): Double {
    var u: Double
    var v: Double
    var s: Double
    do {
        u = nextDouble(-1.0, 1.0)
        v = nextDouble(-1.0, 1.0)
        s = u * u + v * v
    } while (s >= 1 || s == 0.0)
    val multiplier = kotlin.math.sqrt(-2.0 * kotlin.math.ln(s) / s)
    return u * multiplier
}
