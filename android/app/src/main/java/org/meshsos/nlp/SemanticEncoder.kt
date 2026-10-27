package org.meshsos.nlp

import java.util.Locale

/**
 * On-device rule encoder. Kotlin port of nlp/rules.py - keep the two in step.
 *
 * No model file, no accelerator, no network. Runs in microseconds on any
 * handset, which is why it is the baseline a learned encoder has to beat.
 *
 * Handles English and romanised Hindi, because code-mixed input is what the
 * deployment context actually produces.
 */
class SemanticEncoder(private val threshold: Float = DEFAULT_THRESHOLD) {

    data class Result(
        val envelope: SemanticEnvelope,
        val confidence: Float,
        val reasons: List<String>,
        val threshold: Float = DEFAULT_THRESHOLD,
    ) {
        val shouldCompress: Boolean get() = confidence >= threshold
    }

    fun encode(raw: String): Result {
        val text = raw.lowercase(Locale.ROOT).replace(Regex("\\s+"), " ").trim()

        val (category, cTerms) = matchCategory(text)
        val (casualty, nTerms) = matchCasualty(text)
        val (mobility, mTerms) = matchMobility(text)
        val (hazards, hTerms) = matchHazards(text)
        val (severity, sTerms) = matchSeverity(text)

        val matched = cTerms + nTerms + mTerms + hTerms + sTerms
        val (conf, reasons) = score(text, category, casualty, mobility, hazards, severity, matched)

        return Result(
            envelope = SemanticEnvelope(
                category = category,
                casualtyCount = casualty,
                mobility = mobility,
                hazardFlags = hazards,
                severity = severity,
                confidence = Math.round(conf * 7).coerceIn(0, 7),
            ),
            confidence = conf,
            reasons = reasons,
            threshold = threshold,
        )
    }

    // --- field extraction -------------------------------------------------

    private fun hits(text: String, terms: List<String>): List<String> =
        terms.filter { text.contains(it) && !negated(text, it) }

    /**
     * English negates before the term ("no injuries"); romanised Hindi after
     * it ("ghayal nahi"). Both windows are checked; the trailing one is kept
     * short so a later "nahi" cannot reach back and cancel another word.
     */
    private fun negated(text: String, term: String, before: Int = 18, after: Int = 14): Boolean {
        var idx = text.indexOf(term)
        if (idx < 0) return false
        while (idx >= 0) {
            val prefix = text.substring(maxOf(0, idx - before), idx)
            val end = idx + term.length
            val suffix = text.substring(minOf(end, text.length), minOf(end + after, text.length))
            val isNeg = NEGATION_LEADING.any { prefix.contains(it) } ||
                        NEGATION_TRAILING.any { suffix.contains(it) }
            if (!isNeg) return false
            idx = text.indexOf(term, idx + 1)
        }
        return true
    }

    private fun matchCategory(text: String): Pair<Category, List<String>> {
        val found = linkedMapOf<Category, List<String>>()
        for ((cat, terms) in CATEGORY_TERMS) {
            val m = hits(text, terms)
            if (m.isNotEmpty()) found[cat] = m
        }
        for (cat in CATEGORY_PRIORITY) found[cat]?.let { return cat to it }
        return Category.OTHER to emptyList()
    }

    private fun matchCasualty(text: String): Pair<Int, List<String>> {
        var best = -1
        val matched = mutableListOf<String>()

        Regex("\\b(\\d{1,3})\\b").findAll(text).forEach { m ->
            val tail = text.substring(m.range.last + 1, minOf(text.length, m.range.last + 29))
            if (PERSON_WORDS.any { tail.contains(it) }) {
                best = maxOf(best, m.groupValues[1].toInt())
                matched += m.value
            }
        }
        if (best < 0) {
            for ((word, value) in NUMBER_WORDS) {
                val m = Regex("\\b${Regex.escape(word)}\\b").find(text) ?: continue
                val tail = text.substring(m.range.last + 1, minOf(text.length, m.range.last + 29))
                if (PERSON_WORDS.any { tail.contains(it) }) {
                    best = maxOf(best, value)
                    matched += word
                }
            }
        }
        return if (best < 0) 0 to emptyList() else minOf(best, CASUALTY_MANY) to matched
    }

    private fun matchMobility(text: String): Pair<Mobility, List<String>> {
        for (mob in listOf(Mobility.UNRESPONSIVE, Mobility.TRAPPED, Mobility.AMBULATORY)) {
            val m = hits(text, MOBILITY_TERMS.getValue(mob))
            if (m.isNotEmpty()) return mob to m
        }
        return Mobility.UNKNOWN to emptyList()
    }

    private fun matchHazards(text: String): Pair<Int, List<String>> {
        var flags = 0
        val matched = mutableListOf<String>()
        for ((flag, terms) in HAZARD_TERMS) {
            val m = hits(text, terms)
            if (m.isNotEmpty()) { flags = flags or flag; matched += m }
        }
        return flags to matched
    }

    private fun matchSeverity(text: String): Pair<Severity, List<String>> {
        SEVERITY_NONE.firstOrNull { text.contains(it) }?.let { return Severity.NONE to listOf(it) }
        hits(text, SEVERITY_SERIOUS).takeIf { it.isNotEmpty() }?.let { return Severity.SERIOUS to it }
        hits(text, SEVERITY_MINOR).takeIf { it.isNotEmpty() }?.let { return Severity.MINOR to it }
        return Severity.UNKNOWN to emptyList()
    }

    // --- confidence -------------------------------------------------------

    private fun score(
        text: String,
        category: Category,
        casualty: Int,
        mobility: Mobility,
        hazards: Int,
        severity: Severity,
        matched: List<String>,
    ): Pair<Float, List<String>> {
        val reasons = mutableListOf<String>()

        val extras = listOf(
            casualty > 0,
            mobility != Mobility.UNKNOWN,
            hazards != 0,
            severity != Severity.UNKNOWN,
        ).count { it }

        // Confidence asks "did I understand what this message says", not
        // "did I fill all five slots". Scoring filled/5 treats a complete
        // short report as four fifths missing; measured on 10,398 real
        // disaster messages that refused all but 0.2% of them, because
        // 92.8% of real messages carry one field or none -- people write
        // one thing at a time ("we need water", "my house fell"). Those
        // are not partial reports. Keep this in step with nlp/rules.py.
        var s: Float
        if (category != Category.OTHER) {
            s = 0.50f + 0.10f * extras
            reasons += "category recognised, $extras supporting field(s)"
        } else {
            s = 0.12f * extras
            reasons += "no category matched, $extras other field(s)"
        }

        val tokens = Regex("[a-z0-9']+").findAll(text).map { it.value }.filter { it.length > 2 }.toSet()
        val covered = matched.flatMap { Regex("[a-z0-9']+").findAll(it).map { m -> m.value }.toList() }.toSet()
        val unrepresented =
            if (tokens.isEmpty()) 0.0f else 1.0f - tokens.count { it in covered }.toFloat() / tokens.size

        // A message carrying much the encoder could not read is a reason to
        // doubt the compression, not to trust the fields it did find.
        if (unrepresented > 0.85f && tokens.size > 8) {
            s *= 0.6f; reasons += "most of the message was not recognised"
        }
        if (UNCERTAINTY_TERMS.any { text.contains(it) }) {
            s *= 0.8f; reasons += "writer signalled uncertainty"
        }
        if (tokens.size <= 2) {
            s *= 0.7f; reasons += "message too short to judge"
        }
        return s.coerceIn(0.0f, 1.0f) to reasons
    }

    companion object {
        const val DEFAULT_THRESHOLD = 0.45f

        private val NEGATION_LEADING = listOf("no ", "not ", "without ", "nobody ", "none ", "koi nahi")
        private val NEGATION_TRAILING = listOf("nahi", "nhi", "mat ", "na hai")

        private val CATEGORY_TERMS: Map<Category, List<String>> = mapOf(
            Category.TRAPPED to listOf("trapped", "stuck", "pinned", "buried", "under rubble",
                "can't get out",
                "cannot get out", "fansa", "phansa", "fanse", "phanse", "dabba", "dabe",
                "malbe ke niche", "malbe ke neeche", "atka", "atke"),
            Category.COLLAPSE to listOf("collapse", "collapsed", "building fell", "wall fell",
                "roof fell", "rubble", "debris", "girr gaya", "gir gayi", "malba", "malbe",
                "chhat gir", "deewar gir", "imarat gir",
                // from the dev half of the Figure Eight corpus
                "destroyed", "cracked", "crushed"),
            Category.MEDICAL to listOf("injured", "injury", "bleeding", "unconscious", "heart attack",
                "fracture", "broken leg", "broken arm", "medical", "doctor", "ambulance", "stretcher",
                "ghayal", "chot", "khoon", "behosh", "dawai", "bimar", "saans",
                // from the dev half of the Figure Eight corpus
                "wounded", "sick", "hospital", "clinic"),
            Category.FLOOD to listOf("flood", "flooded", "water rising", "drowning", "submerged",
                "waterlogged", "baadh", "badh", "paani bhar", "pani bhar", "doob", "dub rahe"),
            Category.FIRE to listOf("fire", "burning", "smoke", "flames", "blaze",
                "aag", "jal raha", "jal rahi", "dhuan", "dhuaan"),
            Category.SUPPLIES_NEEDED to listOf("need water", "need food", "no water", "no food",
                "supplies", "medicine", "blanket", "khana", "paani chahiye", "pani chahiye",
                "bhookh", "dawa chahiye",
                // Shelter as a thing needed belongs here, not under EVACUATION:
                // the annotation guide defines SUPPLIES_NEEDED as "food, water,
                // medicine, shelter needed". With "shelter" under the
                // higher-priority EVACUATION, "TENT, SHELTER, FOOD AND OTHER
                // THINGS" was read as an evacuation.
                "shelter", "shelter chahiye",
                // Items people actually ask for, from the dev half of the
                // Figure Eight corpus. "tant"/"tants" is how tent gets
                // misspelled; "prelate"/"prelat" is Creole for a tarpaulin.
                "tent", "tant", "prelate", "prelat", "tarpaulin",
                "clothing", "clothes", "blankets", "canned", "batteries", "diapers",
                "hygiene", "hungry", "thirsty", "starving", "nothing to eat"),
            Category.EVACUATION to listOf("evacuate", "evacuation", "need to leave", "get out",
                "nikalna", "nikalo", "bahar nikal"),
            Category.RESCUE_NEEDED to listOf("rescue", "help us", "save us", "send help",
                "need help", "bachao", "madad", "help chahiye", "koi hai"),
        )

        private val CATEGORY_PRIORITY = listOf(
            Category.TRAPPED, Category.COLLAPSE, Category.MEDICAL, Category.FIRE,
            Category.FLOOD, Category.EVACUATION, Category.SUPPLIES_NEEDED,
            Category.RESCUE_NEEDED, Category.OTHER,
        )

        private val MOBILITY_TERMS: Map<Mobility, List<String>> = mapOf(
            Mobility.UNRESPONSIVE to listOf("unconscious", "not responding", "unresponsive",
                "no pulse", "not breathing", "behosh", "jawab nahi", "saans nahi", "hosh nahi"),
            Mobility.TRAPPED to listOf("trapped", "stuck", "pinned", "buried", "can't move",
                "cannot move", "can't get out", "cannot get out", "immobile",
                "fansa", "phansa", "fanse", "phanse", "hil nahi", "nikal nahi", "dabba", "dabe", "atka"),
            Mobility.AMBULATORY to listOf("can walk", "walking", "we are walking", "able to move",
                "mobile", "chal sakte", "chal raha", "chal rahe", "theek hai", "safe hai"),
        )

        private val HAZARD_TERMS: Map<Int, List<String>> = mapOf(
            Hazard.FIRE to listOf("fire", "burning", "flames", "smoke", "aag", "jal raha", "dhuan", "dhuaan"),
            Hazard.WATER to listOf("flood", "flooded", "drowning", "submerged", "rising water",
                "water rising", "waterlogged", "under water", "baadh", "badh",
                "paani bhar", "pani bhar", "doob", "dub rahe"),
            Hazard.GAS to listOf("gas", "gas leak", "lpg", "cylinder", "fumes", "gais"),
            Hazard.STRUCTURAL to listOf("collapse", "collapsed", "rubble", "debris", "unstable",
                "cracks", "wall", "roof", "ceiling", "malba", "malbe", "chhat", "deewar", "darar"),
            Hazard.ELECTRICAL to listOf("live wire", "electric", "electrocut", "short circuit",
                "current", "bijli", "taar", "karent"),
            Hazard.CHEMICAL to listOf("chemical", "toxic", "acid", "ammonia", "fertilizer",
                "pesticide", "zeharila", "rasayan"),
        )

        private val SEVERITY_SERIOUS = listOf("severe", "serious", "critical", "heavy bleeding",
            "unconscious", "not breathing", "dying", "life threatening", "amputat", "deep wound",
            "head injury", "gambhir", "bahut chot", "khoon beh", "khoon bah", "marne", "behosh", "saans nahi")
        private val SEVERITY_MINOR = listOf("minor", "small cut", "scratch", "bruise", "mild",
            "slight", "stable", "halka", "halki", "mamuli", "thodi chot", "theek hun", "theek hoon")
        private val SEVERITY_NONE = listOf("no injury", "no injuries", "nobody hurt", "no one hurt",
            "all safe", "unhurt", "koi chot nahi", "sab theek", "sab safe", "koi ghayal nahi")

        private val NUMBER_WORDS: Map<String, Int> = linkedMapOf(
            "one" to 1, "single" to 1, "ek" to 1,
            "two" to 2, "couple" to 2, "both" to 2, "do" to 2,
            "three" to 3, "teen" to 3, "tin" to 3,
            "four" to 4, "char" to 4, "chaar" to 4,
            "five" to 5, "paanch" to 5, "panch" to 5,
            "six" to 6, "chhe" to 6, "che" to 6,
            "seven" to 7, "saat" to 7, "sat" to 7,
            "eight" to 8, "aath" to 8,
            "nine" to 9, "nau" to 9,
            "ten" to 10, "das" to 10, "dus" to 10,
            "dozen" to 12, "many" to CASUALTY_MANY, "several" to 3,
            "kai" to 3, "bahut" to CASUALTY_MANY, "sab" to CASUALTY_MANY,
        )

        private val PERSON_WORDS = listOf("people", "person", "persons", "men", "women", "children",
            "child", "kids", "kid", "family", "families", "adults", "boys", "girls", "students",
            "workers", "injured", "trapped", "casualt", "victims", "survivors",
            "log", "logo", "admi", "aadmi", "bacche", "baccha", "bache", "aurat", "aurte",
            "parivar", "ladke", "ladki", "jan")

        private val UNCERTAINTY_TERMS = listOf("maybe", "not sure", "i think", "possibly", "unclear",
            "unknown", "approximately", "around", "about", "?", "shayad", "pata nahi",
            "lagta hai", "ho sakta")
    }
}
