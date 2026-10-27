package org.meshsos.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.*
import org.meshsos.battery.BatteryManager
import org.meshsos.crypto.KeyManager
import org.meshsos.gateway.GatewayManager
import org.meshsos.privacy.PrivacyManager
import org.meshsos.protocol.*
import org.meshsos.relay.PriorityLevel
import org.meshsos.relay.RelayCore
import org.meshsos.transport.BleTransport
import org.meshsos.transport.MultiTransportManager
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Foreground Service running MeshSOS mesh radio and relay logic.
 * Declares android:foregroundServiceType="connectedDevice" for Android 14+ compatibility.
 */
class MeshForegroundService : Service() {

    private val CHANNEL_ID = "meshsos_channel"
    private val NOTIFICATION_ID = 911

    private lateinit var relayCore: RelayCore
    private lateinit var transportManager: MultiTransportManager
    private lateinit var batteryManager: BatteryManager
    private lateinit var gatewayManager: GatewayManager
    private lateinit var privacyManager: PrivacyManager
    private lateinit var keyManager: KeyManager

    private val scheduler = Executors.newSingleThreadScheduledExecutor()
    private var wakeLock: PowerManager.WakeLock? = null
    private var currentSeq = 1

    private var fusedClient: FusedLocationProviderClient? = null
    private var locationCallback: LocationCallback? = null
    private var locListener: android.location.LocationListener? = null
    @Volatile
    private var lastKnownLocation: Pair<Double, Double>? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        val notification = buildForegroundNotification("MeshSOS active: scanning for peer beacons")
        startForeground(NOTIFICATION_ID, notification)

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MeshSOS::ServiceWakeLock").apply {
            setReferenceCounted(false)
            acquire()
        }

        privacyManager = PrivacyManager(this)
        batteryManager = BatteryManager(this)
        gatewayManager = GatewayManager(this)
        keyManager = KeyManager()

        initLocationTracking()

        val myNodeId = privacyManager.getOrRotateNodeId()
        relayCore = RelayCore(myNodeId)

        // Initialize BLE transport
        val bleTransport = BleTransport(this)
        transportManager = MultiTransportManager(listOf(bleTransport)) { msg, sender, rssi ->
            handleInboundMessage(msg, sender, rssi)
        }
        transportManager.startAll()

        // Schedule periodic heartbeat & relay queue processing (5s initial delay to allow GPS fix)
        scheduler.scheduleWithFixedDelay({ runTick() }, 5, 5, TimeUnit.SECONDS)
    }

    private fun initLocationTracking() {
        // 1. Google Play Services Fused Location (Cell Towers + Wi-Fi Hotspots + GPS)
        try {
            fusedClient = LocationServices.getFusedLocationProviderClient(this)

            // Meshtastic pattern: Immediate active location query on start
            fusedClient?.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)?.addOnSuccessListener { loc ->
                if (loc != null && (loc.latitude != 0.0 || loc.longitude != 0.0)) {
                    updateCoordinates(loc.latitude, loc.longitude)
                }
            }
            fusedClient?.lastLocation?.addOnSuccessListener { loc ->
                if (loc != null && (loc.latitude != 0.0 || loc.longitude != 0.0)) {
                    updateCoordinates(loc.latitude, loc.longitude)
                }
            }

            // Meshtastic pattern: minUpdateDistanceMeters = 0.0f (ensures stationary indoor devices receive GPS)
            val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 2000L)
                .setMinUpdateIntervalMillis(1000L)
                .setMinUpdateDistanceMeters(0.0f)
                .build()

            locationCallback = object : LocationCallback() {
                override fun onLocationResult(result: LocationResult) {
                    val loc = result.lastLocation ?: return
                    if (loc.latitude != 0.0 || loc.longitude != 0.0) {
                        updateCoordinates(loc.latitude, loc.longitude)
                    }
                }
            }
            fusedClient?.requestLocationUpdates(locationRequest, locationCallback!!, android.os.Looper.getMainLooper())
        } catch (e: SecurityException) {
        } catch (e: Exception) {}

        // 2. Hardware AOSP LocationManager (Fallback)
        try {
            val lm = getSystemService(Context.LOCATION_SERVICE) as? android.location.LocationManager ?: return
            val locGps = lm.getLastKnownLocation(android.location.LocationManager.GPS_PROVIDER)
            val locNet = lm.getLastKnownLocation(android.location.LocationManager.NETWORK_PROVIDER)
            val locPass = lm.getLastKnownLocation(android.location.LocationManager.PASSIVE_PROVIDER)
            val best = locGps ?: locNet ?: locPass
            if (best != null && (best.latitude != 0.0 || best.longitude != 0.0)) {
                updateCoordinates(best.latitude, best.longitude)
            }

            val listener = object : android.location.LocationListener {
                override fun onLocationChanged(loc: android.location.Location) {
                    if (loc.latitude != 0.0 || loc.longitude != 0.0) {
                        updateCoordinates(loc.latitude, loc.longitude)
                    }
                }
                override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                override fun onProviderEnabled(provider: String) {}
                override fun onProviderDisabled(provider: String) {}
            }
            locListener = listener

            // 0.0f distance interval delivers coordinates even while sitting stationary
            if (lm.isProviderEnabled(android.location.LocationManager.GPS_PROVIDER)) {
                lm.requestLocationUpdates(android.location.LocationManager.GPS_PROVIDER, 2000L, 0.0f, listener, android.os.Looper.getMainLooper())
            }
            if (lm.isProviderEnabled(android.location.LocationManager.NETWORK_PROVIDER)) {
                lm.requestLocationUpdates(android.location.LocationManager.NETWORK_PROVIDER, 2000L, 0.0f, listener, android.os.Looper.getMainLooper())
            }
        } catch (e: Exception) {}
    }

    private fun updateCoordinates(lat: Double, lon: Double) {
        if (lat == 0.0 && lon == 0.0) return // Never cache (0,0)
        lastKnownLocation = Pair(lat, lon)
        gatewayManager.updateLocation(lat, lon)
        getSharedPreferences("meshsos_config", Context.MODE_PRIVATE).edit()
            .putFloat("cached_lat", lat.toFloat())
            .putFloat("cached_lon", lon.toFloat())
            .apply()
    }

    companion object {
        /** Neighbours currently heard, for the activity's status line. */
        @Volatile
        @JvmStatic
        var peerCount: Int = 0

        /** SOS messages received from peers, for the status line. */
        @Volatile
        @JvmStatic
        var sosHeard: Int = 0

        @Volatile
        @JvmStatic
        var lastSosFrom: Long = 0L

        // Compact-SOS grid is relative to this origin; both ends must agree.
        // Configuration, distributed at enrolment, never sent per message.
        const val GRID_ORIGIN_LAT = 28.60
        const val GRID_ORIGIN_LON = 77.20

        const val ACTION_SEND_SOS = "org.meshsos.ACTION_SEND_SOS"
        const val ACTION_SEND_OK_NOW = "org.meshsos.ACTION_SEND_OK_NOW"
    }

    private var lastHeartbeatMs: Long = 0L
    private val semanticEncoder = org.meshsos.nlp.SemanticEncoder()

    /**
     * Shared key for the compact SOS auth tag.
     *
     * A per-deployment key provisioned at enrolment belongs here. This
     * placeholder keeps the demo self-contained and MUST NOT ship: anyone
     * with the APK can forge a tag.
     */
    private fun gatewaySharedKey(): ByteArray = "meshsos-gateway-key".toByteArray()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        if (action == ACTION_SEND_SOS) {
            val text = intent.getStringExtra("sos_text") ?: "EMERGENCY HELP NEEDED"
            val (currLat, currLon) = getBestLocation()
            gatewayManager.updateLocation(currLat, currLon)
            val lat = intent.getFloatExtra("lat", currLat.toFloat())
            val lon = intent.getFloatExtra("lon", currLon.toFloat())
            val seq = intent.getIntExtra("sos_seq", nextSeq())
            val nowMs = System.currentTimeMillis()

            // Compress on-device first. If the encoder is confident the whole
            // report fits one 25-byte advertising frame; if it is not, the
            // original text goes out uncompressed rather than a guess.
            val encoded = semanticEncoder.encode(text)
            val grid = org.meshsos.protocol.CompactSosMessage.latLonToGrid(
                lat.toDouble(), lon.toDouble(), GRID_ORIGIN_LAT, GRID_ORIGIN_LON
            )
            if (encoded.shouldCompress && grid != null) {
                val compact = org.meshsos.protocol.CompactSosMessage(
                    header = Header(
                        org.meshsos.protocol.CompactSosMessage.VERSION,
                        MessageType.SOS, relayCore.nodeId, seq, 8, nowMs / 1000L
                    ),
                    gridX = grid.first,
                    gridY = grid.second,
                    battery = batteryManager.getBatteryPercent(),
                    envelope = encoded.envelope,
                )
                compact.sign(gatewaySharedKey())
                transportManager.send(compact, PriorityLevel.SOS)
                if (gatewayManager.checkUplinkAvailable()) {
                    gatewayManager.queueForUplink(compact)
                    scheduler.execute { flushGateway() }
                }
                return START_STICKY
            }

            val dummyPub = ByteArray(32) { 0x42 }
            val sos = SosMessage(
                header = Header(1, MessageType.SOS, relayCore.nodeId, seq, 8, nowMs / 1000L),
                lat = lat,
                lon = lon,
                accuracy = 5.0f,
                category = SosCategory.OTHER,
                battery = batteryManager.getBatteryPercent(),
                text = text,
                pubkey = dummyPub,
                signature = ByteArray(64) { 0x01 }
            )
            try {
                sos.signature = keyManager.signData(sos.unsignedPayloadBytes())
            } catch (e: Exception) {}

            transportManager.send(sos, PriorityLevel.SOS)
            if (gatewayManager.checkUplinkAvailable()) {
                gatewayManager.queueForUplink(sos)
                scheduler.execute { flushGateway() }
            }
        } else if (action == ACTION_SEND_OK_NOW) {
            val targetSeq = intent.getIntExtra("target_seq", 1)
            val nowMs = System.currentTimeMillis()
            val dummyPub = ByteArray(32) { 0x42 }
            val ok = OkNowMessage(
                header = Header(1, MessageType.OK_NOW, relayCore.nodeId, nextSeq(), 8, nowMs / 1000L),
                targetSosSeq = targetSeq,
                battery = batteryManager.getBatteryPercent(),
                text = "I'm OK now",
                pubkey = dummyPub,
                signature = ByteArray(64) { 0x01 }
            )
            try {
                ok.signature = keyManager.signData(ok.unsignedPayloadBytes())
            } catch (e: Exception) {}

            transportManager.send(ok, PriorityLevel.OK_NOW)
            if (gatewayManager.checkUplinkAvailable()) {
                gatewayManager.queueForUplink(ok)
                scheduler.execute { flushGateway() }
            }
        }
        return START_STICKY
    }

    private fun handleInboundMessage(message: MeshMessage, senderId: Long, rssi: Float) {
        val nowMs = System.currentTimeMillis()
        val queued = relayCore.onReceive(message, nowMs, randomFraction = Math.random())

        // Receiving an SOS was entirely invisible: the only notification
        // this service posts is the static "scanning for peer beacons"
        // one. On a two-phone test the receiving handset showed nothing
        // at all, which is indistinguishable from the sender having
        // failed. Heartbeats stay silent; an SOS does not.
        if (message is CompactSosMessage || message is SosMessage) {
            sosHeard += 1
            lastSosFrom = senderId
            val what = if (message is CompactSosMessage) {
                val e = message.envelope
                "${e.category.name.replace('_', ' ').lowercase()}" +
                    (if (e.casualtyCount > 0) ", ${e.casualtyCount} people" else "")
            } else {
                "full text"
            }
            updateNotification(
                "SOS heard from ${senderId and 0xFFFF}: $what  " +
                "(${sosHeard} total, ${rssi.toInt()} dBm)"
            )
        }

        // As soon as a peer message is received via BLE, if this phone has internet, immediately forward to backend!
        if (gatewayManager.checkUplinkAvailable()) {
            gatewayManager.queueForUplink(message)
            scheduler.execute { flushGateway() }
        }
    }

    private fun flushGateway() {
        if (!gatewayManager.checkUplinkAvailable()) return
        val nowMs = System.currentTimeMillis()
        val acks = gatewayManager.flushBatchToBackend(relayCore.nodeId)
        acks.forEach { ack ->
            relayCore.onReceive(ack, nowMs)
            transportManager.send(ack, PriorityLevel.SOS)
        }
    }

    private fun runTick() {
        val nowMs = System.currentTimeMillis()
        val batteryPct = batteryManager.getBatteryPercent()

        // 1. Critical Battery Check
        if (batteryManager.checkCriticalBattery(batteryPct)) {
            val lowbat = LowbatMessage(
                header = Header(1, MessageType.LOWBAT, relayCore.nodeId, nextSeq(), 5, nowMs / 1000L),
                reason = LowbatReason.BATTERY_DEPLETED,
                battery = batteryPct,
                lastGridX = 0,
                lastGridY = 0
            )
            transportManager.send(lowbat, PriorityLevel.LOWBAT)
            stopSelf()
            return
        }

        // 2. Transmit periodic heartbeat if interval elapsed
        val neighbours = transportManager.getAllNeighbours()
        peerCount = neighbours.size
        val trickleInterval = batteryManager.updateTrickleInterval(neighbours.keys)

        if (nowMs - lastHeartbeatMs >= trickleInterval || lastHeartbeatMs == 0L) {
            lastHeartbeatMs = nowMs
            val digest = neighbours.keys.take(2).map { (it and 0xFFFF).toInt() }
            val (currLat, currLon) = getBestLocation()
            gatewayManager.updateLocation(currLat, currLon)
            val (gx, gy) = privacyManager.quantizeToGridCell(currLat, currLon)
            val hb = HeartbeatMessage(
                header = Header(1, MessageType.HEARTBEAT, relayCore.nodeId, nextSeq(), 8, nowMs / 1000L),
                gridX = gx,
                gridY = gy,
                battery = batteryPct,
                flags = 0,
                neighbours = digest
            )
            transportManager.send(hb, PriorityLevel.HEARTBEAT)
            if (gatewayManager.checkUplinkAvailable()) {
                gatewayManager.queueForUplink(hb)
            }
        }

        // Check and pop any matured relay transmissions
        var readyMsg = relayCore.popReadyTransmission(nowMs)
        while (readyMsg != null) {
            val prio = PriorityLevel.fromType(readyMsg.header.type)
            transportManager.send(readyMsg, prio)
            readyMsg = relayCore.popReadyTransmission(nowMs)
        }

        // 3. Flush gateway batch if connected
        flushGateway()
    }

    private fun nextSeq(): Int {
        currentSeq = (currentSeq + 1) and 0xFFFF
        if (currentSeq == 0) currentSeq = 1
        return currentSeq
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "MeshSOS Emergency Network",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Keeps peer-to-peer disaster mesh alive with low-power BLE presence"
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun updateNotification(text: String) {
        try {
            val nm = getSystemService(NotificationManager::class.java)
            nm.notify(NOTIFICATION_ID, buildForegroundNotification(text))
        } catch (e: Exception) {
            // A notification is a convenience here; never let it take the
            // service down while it is relaying.
        }
    }

    private fun buildForegroundNotification(statusText: String): Notification {
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("MeshSOS Active")
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setOngoing(true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            builder.setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
        }
        return builder.build()
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)
        try {
            val restartIntent = Intent(applicationContext, MeshForegroundService::class.java).also {
                it.setPackage(packageName)
            }
            val pendingIntent = PendingIntent.getService(
                this, 1, restartIntent,
                PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
            )
            val alarmService = getSystemService(Context.ALARM_SERVICE) as? AlarmManager
            alarmService?.set(
                AlarmManager.ELAPSED_REALTIME,
                android.os.SystemClock.elapsedRealtime() + 1000,
                pendingIntent
            )
        } catch (e: Exception) {}
    }

    private fun getBestLocation(): Pair<Double, Double> {
        lastKnownLocation?.let { if (it.first != 0.0 || it.second != 0.0) return it }
        try {
            val lm = getSystemService(Context.LOCATION_SERVICE) as? android.location.LocationManager
            if (lm != null) {
                val locGps = lm.getLastKnownLocation(android.location.LocationManager.GPS_PROVIDER)
                val locNet = lm.getLastKnownLocation(android.location.LocationManager.NETWORK_PROVIDER)
                val locPass = lm.getLastKnownLocation(android.location.LocationManager.PASSIVE_PROVIDER)
                val best = locGps ?: locNet ?: locPass
                if (best != null) {
                    val pair = Pair(best.latitude, best.longitude)
                    lastKnownLocation = pair
                    return pair
                }
            }
        } catch (e: SecurityException) {
            // Fallback when permission or sensor pending
        }

        // Try cached coords from previous run
        val prefs = getSharedPreferences("meshsos_config", Context.MODE_PRIVATE)
        val cachedLat = prefs.getFloat("cached_lat", 0f).toDouble()
        val cachedLon = prefs.getFloat("cached_lon", 0f).toDouble()
        if (cachedLat != 0.0 && cachedLon != 0.0) {
            return Pair(cachedLat, cachedLon)
        }

        return Pair(0.0, 0.0)
    }

    override fun onDestroy() {
        super.onDestroy()
        locationCallback?.let { fusedClient?.removeLocationUpdates(it) }
        locListener?.let {
            val lm = getSystemService(Context.LOCATION_SERVICE) as? android.location.LocationManager
            try { lm?.removeUpdates(it) } catch (e: Exception) {}
        }
        scheduler.shutdown()
        transportManager.stopAll()
        wakeLock?.let { if (it.isHeld) it.release() }
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
