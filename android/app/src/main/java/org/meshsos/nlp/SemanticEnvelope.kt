package org.meshsos.nlp

/**
 * A free-text emergency report compressed to 3 bytes, so a complete
 * authenticated SOS fits one legacy BLE advertising frame.
 *
 * Layout (24 bits, big-endian) - must stay byte-identical to nlp/envelope.py:
 *
 *   bits 23..20  category        (4)
 *   bits 19..17  casualtyCount   (3)   0-6 people, 7 = "7 or more"
 *   bits 16..15  mobility        (2)
 *   bits 14..9   hazardFlags     (6)
 *   bits  8..7   severity        (2)
 *   bits  6..4   confidence      (3)
 *   bits  3..0   reserved        (4)
 */

enum class Category(val code: Int) {
    OTHER(0), MEDICAL(1), TRAPPED(2), FLOOD(3), COLLAPSE(4),
    FIRE(5), RESCUE_NEEDED(6), SUPPLIES_NEEDED(7), EVACUATION(8);

    companion object {
        fun fromCode(c: Int) = values().firstOrNull { it.code == c } ?: OTHER
    }
}

enum class Mobility(val code: Int) {
    UNKNOWN(0), AMBULATORY(1), TRAPPED(2), UNRESPONSIVE(3);

    companion object {
        fun fromCode(c: Int) = values().firstOrNull { it.code == c } ?: UNKNOWN
    }
}

enum class Severity(val code: Int) {
    UNKNOWN(0), NONE(1), MINOR(2), SERIOUS(3);

    companion object {
        fun fromCode(c: Int) = values().firstOrNull { it.code == c } ?: UNKNOWN
    }
}

object Hazard {
    const val FIRE = 1
    const val WATER = 2
    const val GAS = 4
    const val STRUCTURAL = 8
    const val ELECTRICAL = 16
    const val CHEMICAL = 32

    val NAMES = linkedMapOf(
        FIRE to "fire",
        WATER to "water",
        GAS to "gas",
        STRUCTURAL to "structural collapse",
        ELECTRICAL to "live electricity",
        CHEMICAL to "chemicals",
    )
}

const val ENVELOPE_BYTES = 3
const val CASUALTY_MANY = 7

data class SemanticEnvelope(
    val category: Category = Category.OTHER,
    val casualtyCount: Int = 0,
    val mobility: Mobility = Mobility.UNKNOWN,
    val hazardFlags: Int = 0,
    val severity: Severity = Severity.UNKNOWN,
    val confidence: Int = 0,
) {
    init {
        require(casualtyCount in 0..7) { "casualtyCount must be 0-7, got $casualtyCount" }
        require(hazardFlags in 0..63) { "hazardFlags must be 0-63, got $hazardFlags" }
        require(confidence in 0..7) { "confidence must be 0-7, got $confidence" }
    }

    fun pack(): ByteArray {
        val v = ((category.code and 0xF) shl 20) or
                ((casualtyCount and 0x7) shl 17) or
                ((mobility.code and 0x3) shl 15) or
                ((hazardFlags and 0x3F) shl 9) or
                ((severity.code and 0x3) shl 7) or
                ((confidence and 0x7) shl 4)
        return byteArrayOf(
            ((v shr 16) and 0xFF).toByte(),
            ((v shr 8) and 0xFF).toByte(),
            (v and 0xFF).toByte(),
        )
    }

    fun confidenceFraction(): Float = confidence / 7.0f

    // Parentheses are load-bearing: Kotlin's infix `and` binds looser than
    // `!=`, so `hazardFlags and it != 0` parses as `hazardFlags and (it != 0)`.
    fun hazards(): List<Int> = Hazard.NAMES.keys.filter { (hazardFlags and it) != 0 }

    /** One line for the local status pill, mirroring nlp/decoder.py. */
    fun describe(): String {
        val parts = mutableListOf(
            when (category) {
                Category.OTHER -> "Emergency"
                Category.MEDICAL -> "Medical emergency"
                Category.TRAPPED -> "People trapped"
                Category.FLOOD -> "Flooding"
                Category.COLLAPSE -> "Structural collapse"
                Category.FIRE -> "Fire"
                Category.RESCUE_NEEDED -> "Rescue requested"
                Category.SUPPLIES_NEEDED -> "Supplies needed"
                Category.EVACUATION -> "Evacuation needed"
            }
        )
        if (casualtyCount > 0) {
            parts += when (casualtyCount) {
                CASUALTY_MANY -> "7 or more people"
                1 -> "1 person"
                else -> "$casualtyCount people"
            }
        }
        when (mobility) {
            Mobility.AMBULATORY -> parts += "can move unaided"
            Mobility.TRAPPED -> parts += "cannot move"
            Mobility.UNRESPONSIVE -> parts += "unresponsive"
            Mobility.UNKNOWN -> {}
        }
        when (severity) {
            Severity.NONE -> parts += "no injuries"
            Severity.MINOR -> parts += "minor injuries"
            Severity.SERIOUS -> parts += "serious injuries"
            Severity.UNKNOWN -> {}
        }
        val hz = hazards().map { Hazard.NAMES.getValue(it) }
        if (hz.isNotEmpty()) parts += "hazard: " + hz.joinToString(", ")
        return parts.joinToString("; ")
    }

    companion object {
        fun unpack(data: ByteArray): SemanticEnvelope {
            require(data.size == ENVELOPE_BYTES) { "envelope must be $ENVELOPE_BYTES bytes" }
            val v = ((data[0].toInt() and 0xFF) shl 16) or
                    ((data[1].toInt() and 0xFF) shl 8) or
                    (data[2].toInt() and 0xFF)
            return SemanticEnvelope(
                category = Category.fromCode((v shr 20) and 0xF),
                casualtyCount = (v shr 17) and 0x7,
                mobility = Mobility.fromCode((v shr 15) and 0x3),
                hazardFlags = (v shr 9) and 0x3F,
                severity = Severity.fromCode((v shr 7) and 0x3),
                confidence = (v shr 4) and 0x7,
            )
        }
    }
}
