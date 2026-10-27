package org.meshsos.protocol

// A sealed class's subclasses must share its package, so this message
// lives beside MeshMessage rather than with the encoder that fills it.
import org.meshsos.nlp.ENVELOPE_BYTES
import org.meshsos.nlp.SemanticEnvelope
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * An authenticated SOS that fits one legacy BLE advertising frame.
 *
 * Must stay byte-identical to protocol/compact_sos.py.
 *
 *   header       13   version=2 marks this as the compact layout
 *   gridX         2
 *   gridY         2
 *   battery       1
 *   envelope      3
 *   authTag       4   truncated HMAC-SHA256
 *   -----------------
 *   total        25   <= 27 usable bytes of an advertising frame
 *
 * The full SosMessage shares message type SOS, so the version byte is what
 * tells the two layouts apart on the wire.
 *
 * The tag is NOT a digital signature. It authenticates to a gateway holding
 * the shared key and does not prove origin to a third party.
 */
data class CompactSosMessage(
    override val header: Header,
    val gridX: Int,
    val gridY: Int,
    val battery: Int,
    val envelope: SemanticEnvelope,
    var authTag: ByteArray = ByteArray(AUTH_TAG_BYTES),
) : MeshMessage(header) {

    init {
        require(header.type == MessageType.SOS) { "CompactSosMessage requires type SOS" }
        require(battery in 0..100) { "battery must be 0-100, got $battery" }
        require(authTag.size == AUTH_TAG_BYTES) { "authTag must be $AUTH_TAG_BYTES bytes" }
    }

    /**
     * Bytes the tag covers. TTL is zeroed: it drops at every hop, so a tag
     * over the live header would verify only at the originating node.
     */
    private fun unauthenticatedBytes(): ByteArray {
        val buf = ByteBuffer.allocate(Header.SIZE + BODY_SIZE + ENVELOPE_BYTES)
            .order(ByteOrder.BIG_ENDIAN)
        header.copy(ttl = 0).pack(buf)
        buf.putShort(gridX.toShort())
        buf.putShort(gridY.toShort())
        buf.put(battery.toByte())
        buf.put(envelope.pack())
        return buf.array()
    }

    fun computeTag(sharedKey: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(sharedKey, "HmacSHA256"))
        return mac.doFinal(unauthenticatedBytes()).copyOf(AUTH_TAG_BYTES)
    }

    fun sign(sharedKey: ByteArray) {
        authTag = computeTag(sharedKey)
    }

    fun verify(sharedKey: ByteArray): Boolean =
        // Constant-time compare. The broadcast medium offers no practical
        // timing oracle, but MessageDigest.isEqual costs nothing over
        // contentEquals and removes the question entirely.
        java.security.MessageDigest.isEqual(computeTag(sharedKey), authTag)

    override fun toBytes(): ByteArray {
        val out = unauthenticatedBytes() + authTag
        check(out.size == SIZE) { "compact SOS must be $SIZE bytes, got ${out.size}" }
        return out
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CompactSosMessage) return false
        return header == other.header && gridX == other.gridX && gridY == other.gridY &&
               battery == other.battery && envelope == other.envelope &&
               authTag.contentEquals(other.authTag)
    }

    override fun hashCode(): Int {
        var r = header.hashCode()
        r = 31 * r + gridX
        r = 31 * r + gridY
        r = 31 * r + battery
        r = 31 * r + envelope.hashCode()
        r = 31 * r + authTag.contentHashCode()
        return r
    }

    companion object {
        const val VERSION = 2
        const val AUTH_TAG_BYTES = 4
        const val BODY_SIZE = 5           // gridX, gridY, battery
        const val SIZE = Header.SIZE + BODY_SIZE + ENVELOPE_BYTES + AUTH_TAG_BYTES  // 25

        /** Usable payload of a legacy advertising frame. docs/DECISIONS.md 1.1. */
        const val ADV_USABLE_PAYLOAD_BYTES = 27

        /** ~100 m cells, relative to a deployment origin set at enrolment. */
        const val GRID_CELL_DEGREES = 0.001

        fun fitsSingleFrame(nbytes: Int) = nbytes <= ADV_USABLE_PAYLOAD_BYTES

        fun latLonToGrid(lat: Double, lon: Double, originLat: Double, originLon: Double): Pair<Int, Int>? {
            val gx = Math.round((lat - originLat) / GRID_CELL_DEGREES).toInt()
            val gy = Math.round((lon - originLon) / GRID_CELL_DEGREES).toInt()
            // Outside the 16-bit window the sender must fall back to a full SOS.
            if (gx !in -32768..32767 || gy !in -32768..32767) return null
            return gx to gy
        }

        /** Continue from a buffer whose header has already been read. */
        fun unpack(header: Header, buffer: ByteBuffer): CompactSosMessage {
            val gridX = buffer.short.toInt()
            val gridY = buffer.short.toInt()
            val battery = buffer.get().toInt() and 0xFF
            val env = ByteArray(ENVELOPE_BYTES).also { buffer.get(it) }
            val tag = ByteArray(AUTH_TAG_BYTES).also { buffer.get(it) }
            return CompactSosMessage(header, gridX, gridY, battery,
                SemanticEnvelope.unpack(env), tag)
        }

        fun fromBytes(data: ByteArray): CompactSosMessage {
            require(data.size == SIZE) { "expected $SIZE bytes, got ${data.size}" }
            val buf = ByteBuffer.wrap(data).order(ByteOrder.BIG_ENDIAN)
            val header = Header.unpack(buf)
            val gridX = buf.short.toInt()
            val gridY = buf.short.toInt()
            val battery = buf.get().toInt() and 0xFF
            val env = ByteArray(ENVELOPE_BYTES).also { buf.get(it) }
            val tag = ByteArray(AUTH_TAG_BYTES).also { buf.get(it) }
            return CompactSosMessage(header, gridX, gridY, battery,
                SemanticEnvelope.unpack(env), tag)
        }
    }
}
