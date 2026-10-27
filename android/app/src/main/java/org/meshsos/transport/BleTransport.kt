package org.meshsos.transport

import android.bluetooth.BluetoothAdapter
import android.bluetooth.le.*
import android.content.Context
import android.os.ParcelUuid
import org.meshsos.protocol.MeshMessage
import org.meshsos.protocol.MeshProtocolDeserializer
import org.meshsos.relay.PriorityLevel
import java.util.*
import java.util.concurrent.ConcurrentHashMap

/**
 * Native Bluetooth Low Energy transport implementation.
 * Incorporates all verified Android BLE constraints from DECISIONS.md:
 * - Fits within 31-byte legacy advertising payload.
 * - Screen-off background scanning requires hardware-filtered scans (ScanFilter with ServiceUuid).
 * - Hardware scan buffering with setReportDelay() prevents Doze drops.
 */
class BleTransport(
    private val context: Context,
    private val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter()
) : Transport {

    override val transportName: String = "BLE_Legacy_Adv"
    private val serviceUuid = ParcelUuid(UUID.fromString("0000FD01-0000-1000-8000-00805F9B34FB"))

    private var advertiser: BluetoothLeAdvertiser? = null
    private var scanner: BluetoothLeScanner? = null
    private var receiveListener: ((MeshMessage, Long, Float) -> Unit)? = null

    // RSSI and the time each neighbour was last heard. Entries used to be
    // added and never removed, so the peer count only ever rose: a phone
    // switched off or carried out of range stayed counted until the
    // service restarted. That makes the count unusable for the one thing
    // a field tester needs it for -- telling whether the other handset is
    // reachable right now.
    private val neighboursMap = ConcurrentHashMap<Long, Float>()
    private val lastHeardMs = ConcurrentHashMap<Long, Long>()
    private var isScanning = false

    override fun start() {
        Status.bluetoothOn = bluetoothAdapter != null && bluetoothAdapter.isEnabled
        if (!Status.bluetoothOn) {
            Status.advertiserReady = false
            Status.lastError = null      // not an error, just switched off
            return
        }
        advertiser = bluetoothAdapter!!.bluetoothLeAdvertiser
        scanner = bluetoothAdapter.bluetoothLeScanner
        Status.advertiserReady = advertiser != null
        if (advertiser == null) {
            Status.lastError = "this phone cannot advertise over BLE"
        }
        startFilteredScanning()
    }

    private val advertiseCallback = object : AdvertiseCallback() {
        override fun onStartSuccess(settingsInEffect: AdvertiseSettings?) {
            Status.advertising = true
            Status.lastError = null
        }

        override fun onStartFailure(errorCode: Int) {
            Status.advertising = false
            // This was an empty method. Every advertising failure -- a
            // missing Nearby devices permission, an oversized payload, a
            // radio already in use -- was discarded here, while the UI
            // went on saying the SOS had been dispatched. On a field test
            // across ten handsets that is indistinguishable from a
            // protocol fault.
            Status.lastError = when (errorCode) {
                ADVERTISE_FAILED_DATA_TOO_LARGE ->
                    "payload too large for one advertisement"
                ADVERTISE_FAILED_TOO_MANY_ADVERTISERS ->
                    "radio busy: another app is advertising"
                ADVERTISE_FAILED_ALREADY_STARTED ->
                    "already advertising"
                ADVERTISE_FAILED_INTERNAL_ERROR ->
                    "Bluetooth internal error, toggle Bluetooth off and on"
                ADVERTISE_FAILED_FEATURE_UNSUPPORTED ->
                    "this phone cannot advertise over BLE"
                else -> "advertising failed, code $errorCode"
            }
        }
    }

    /**
     * What the radio is actually doing, readable from the UI.
     *
     * The transport lives inside the foreground service and the activity
     * cannot reach it, so the few facts a tester needs are kept here.
     * Nothing in the app's behaviour depends on these fields; they exist
     * so a failure is visible instead of silent.
     */
    object Status {
        @Volatile var bluetoothOn: Boolean = false
        @Volatile var advertiserReady: Boolean = false
        @Volatile var advertising: Boolean = false
        @Volatile var scanning: Boolean = false
        @Volatile var lastError: String? = null
        @Volatile var sent: Int = 0
        @Volatile var received: Int = 0
        @Volatile var lastSentBytes: Int = 0

        /** One line for the status badge. */
        fun summary(peers: Int): String {
            val err = lastError
            if (!bluetoothOn) return "Bluetooth is off"
            if (!advertiserReady) return "This phone cannot advertise over BLE"
            if (err != null) return "Radio: $err"
            if (!advertising && sent == 0) return "Radio ready, nothing sent yet"
            return "Sent $sent · heard $received · $peers peer" +
                (if (peers == 1) "" else "s") +
                (if (advertising) " · on air ${lastSentBytes} B" else "")
        }
    }

    override fun stop() {
        stopScanning()
        try {
            advertiser?.stopAdvertising(advertiseCallback)
        } catch (e: Exception) {}
    }

    override fun send(message: MeshMessage, priority: PriorityLevel): Boolean {
        if (advertiser == null) {
            Status.lastError =
                if (!Status.bluetoothOn) "Bluetooth is off"
                else "no BLE advertiser on this phone"
            return false
        }
        val bytes = when (message) {
            is org.meshsos.protocol.SosMessage -> message.toCompactBleBytes()
            is org.meshsos.protocol.OkNowMessage -> message.toCompactBleBytes()
            else -> message.toBytes()
        }
        if (bytes.size > 26) {
            Status.lastError = "message is ${bytes.size} B, one advertisement holds 26"
            return false
        }

        try {
            advertiser?.stopAdvertising(advertiseCallback)
        } catch (e: Exception) {}

        val settings = AdvertiseSettings.Builder()
            .setAdvertiseMode(
                if (priority == PriorityLevel.SOS) AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY
                else AdvertiseSettings.ADVERTISE_MODE_BALANCED
            )
            .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
            .setConnectable(false)
            .setTimeout(0) // Continuous transmission until replaced or stopped
            .build()

        val data = AdvertiseData.Builder()
            .addServiceData(serviceUuid, bytes)
            .setIncludeDeviceName(false)
            .setIncludeTxPowerLevel(false)
            .build()

        return try {
            advertiser?.startAdvertising(settings, data, advertiseCallback)
            Status.sent += 1
            Status.lastSentBytes = bytes.size
            true
        } catch (e: SecurityException) {
            // On Android 12+ this is what a missing BLUETOOTH_ADVERTISE
            // (Nearby devices) permission looks like.
            Status.advertising = false
            Status.lastError = "Nearby devices permission not granted"
            false
        } catch (e: Exception) {
            Status.advertising = false
            Status.lastError = "send failed: ${e.javaClass.simpleName}"
            false
        }
    }

    private fun startFilteredScanning() {
        if (isScanning || scanner == null) return

        // ScanFilter matching our 16-bit expanded service data with mask 0 (matches any payload length >= 1)
        val filterWithData = ScanFilter.Builder()
            .setServiceData(serviceUuid, byteArrayOf(0), byteArrayOf(0))
            .build()
        val emptyFilter = ScanFilter.Builder().build()

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_BALANCED)
            .setReportDelay(0L) // Immediate delivery to eliminate hop latency
            .build()

        try {
            scanner?.startScan(listOf(filterWithData, emptyFilter), settings, scanCallback)
            isScanning = true
            Status.scanning = true
        } catch (e: SecurityException) {
            // Lacking BLUETOOTH_SCAN runtime permission
        } catch (e: Exception) {}
    }

    private fun stopScanning() {
        if (!isScanning) return
        try {
            scanner?.stopScan(scanCallback)
        } catch (e: Exception) {}
        isScanning = false
    }

    private val scanCallback = object : ScanCallback() {
        override fun onBatchScanResults(results: MutableList<ScanResult>?) {
            results?.forEach { processScanResult(it) }
        }

        override fun onScanResult(callbackType: Int, result: ScanResult?) {
            result?.let { processScanResult(it) }
        }

        override fun onScanFailed(errorCode: Int) {}
    }

    private fun processScanResult(result: ScanResult) {
        val record = result.scanRecord ?: return
        val serviceData = record.getServiceData(serviceUuid) ?: return
        try {
            val message = MeshProtocolDeserializer.deserialize(serviceData)
            val senderId = message.header.nodeId
            val rssi = result.rssi.toFloat()
            neighboursMap[senderId] = rssi
            lastHeardMs[senderId] = System.currentTimeMillis()
            Status.received += 1
            receiveListener?.invoke(message, senderId, rssi)
        } catch (e: Exception) {
            // Corrupt or non-protocol packet
        }
    }

    override fun getNeighbours(): Map<Long, Float> {
        // Heartbeats go out every 5 s, so a peer unheard for 30 s has
        // missed six of them and is treated as gone.
        val cutoff = System.currentTimeMillis() - NEIGHBOUR_TIMEOUT_MS
        lastHeardMs.entries.removeIf { it.value < cutoff }
        neighboursMap.keys.retainAll(lastHeardMs.keys)
        return neighboursMap
    }

    companion object {
        const val NEIGHBOUR_TIMEOUT_MS = 30_000L
    }

    override fun getCapabilities(): Set<String> = setOf("BLE_BROADCAST", "LOW_POWER", "FILTERED_SCAN")

    override fun registerReceiveListener(listener: (message: MeshMessage, senderId: Long, rssi: Float) -> Unit) {
        this.receiveListener = listener
    }
}
