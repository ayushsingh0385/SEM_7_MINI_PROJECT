"""Out-of-distribution sanity check for the text detector.

    python scripts/sanity_check.py

Why this exists
---------------
A high test-set score only says the model separates *that dataset*. The first
version of this project scored 99.5% on its held-out split and then labelled
every piece of hand-written prose as "AI-generated" with ~100% confidence,
because the training corpus had been stripped of punctuation and the model had
never seen a capital letter or a comma.

The test split could not catch that. This script can: the passages below were
written by hand for this check and appear in no dataset, so they measure whether
the model generalises rather than whether it memorised a corpus.

Treat a failure here as more serious than a point of test accuracy.
"""
import argparse

import _bootstrap  # noqa: F401

# Hand-written passages. Human ones are deliberately varied in register: formal,
# informal, technical, and short - the range a real user would paste in.
HUMAN_SAMPLES = [
    ("informal diary note", """
ok so i finally got round to fixing the bike -- took way longer than i thought!!
the chain was rusted solid. anyway, 3 hours and a LOT of swearing later, it
works. sort of. the gears still slip on the 4th but honestly? good enough for
now. might take it to the shop next week if i can be bothered. we'll see.
Started around 9am thinking it'd be a quick job. It was not. First the pedal
crank wouldn't come off, then i stripped a bolt, then i had to walk to the
hardware shop because of course i didn't have the right size allen key.
"""),
    ("formal history essay", """
The industrial revolution fundamentally reshaped the economic structure of
Britain between 1760 and 1840. Prior to this period, the majority of the
population was engaged in agricultural labour, with production organised around
small household units. The introduction of mechanised spinning and, later, steam
power displaced this arrangement entirely. Factory production concentrated
workers in urban centres, which in turn created the housing crises documented
extensively by contemporary observers such as Engels. Historians continue to
debate whether living standards rose or fell during the early decades; the
pessimist case rests largely on real wage data, while optimists point to
consumption patterns and the availability of cheap cotton goods.
"""),
    ("technical explanation", """
The reason the query is slow is that there's no index on the created_at column,
so Postgres has to do a sequential scan across all 40 million rows every time.
You can see this in the EXPLAIN output - look for "Seq Scan on events". Adding a
btree index on created_at should drop it to a few milliseconds, but be careful:
the table gets heavy writes, and every index you add slows those down. I'd
measure the write throughput before and after rather than assuming it's fine.
Also worth checking whether the ORM is generating the query you think it is.
"""),
    ("conversational reply", """
Honestly I think you're overthinking it. We tried the same approach last year
and it fell apart for exactly the reason Priya mentioned in standup - the
upstream service doesn't guarantee ordering, so you end up reconciling anyway.
My vote is to keep the simple version, ship it, and revisit if the error rate
actually becomes a problem. It probably won't. We've got bigger fires.
"""),
]

# Written to imitate typical assistant output: even sentence lengths, hedged
# phrasing, structured enumeration, tidy summary sentence.
AI_SAMPLES = [
    ("assistant-style explainer", """
There are several important factors to consider when evaluating this approach.
First, it is essential to understand that performance optimization involves
trade-offs between speed and maintainability. Second, the implementation should
be carefully designed to ensure scalability as the system grows. Additionally,
it is worth noting that proper testing plays a crucial role in maintaining code
quality over time. In conclusion, by carefully considering these factors, teams
can make informed decisions that balance immediate needs with long-term
sustainability goals.
"""),
    ("assistant-style listicle", """
Regular exercise offers numerous benefits for both physical and mental health.
Firstly, it improves cardiovascular function by strengthening the heart muscle
and improving circulation throughout the body. Secondly, physical activity
releases endorphins, which are natural chemicals that help reduce stress and
improve mood. Furthermore, consistent exercise supports better sleep quality and
can enhance cognitive function. It is important to note that individuals should
consult with a healthcare professional before beginning any new fitness regimen.
"""),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", action="store_true",
                   help="show the top style signals for each passage")
    args = p.parse_args()

    from mmdetect.text.infer import get_text_detector

    det = get_text_detector()
    if not det.is_available:
        print("No trained text model. Run: python scripts/train_text.py")
        return
    det.load()

    print("=" * 70)
    print("  OUT-OF-DISTRIBUTION SANITY CHECK  (hand-written, not from any dataset)")
    print("=" * 70)

    correct = 0
    total = 0
    for expected, samples in (("Human", HUMAN_SAMPLES), ("AI", AI_SAMPLES)):
        print(f"\n  Expected: {expected}")
        print("  " + "-" * 66)
        for name, text in samples:
            text = " ".join(text.split())
            r = det.predict(text)
            got_ai = r.ai_probability >= 0.5
            ok = (expected == "AI") == got_ai
            correct += ok
            total += 1
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<28} P(AI)={r.ai_probability:6.3f}  {r.verdict}")
            if args.verbose and r.explanation.get("signals"):
                for s in r.explanation["signals"][:3]:
                    print(f"          {s['label']:<34}{s['value']:>9.3f} -> {s['points_to']}")

    print("\n" + "=" * 70)
    print(f"  {correct}/{total} correct on hand-written text")
    if correct < total:
        print("\n  Passages the model got wrong are the ones worth reading. A model")
        print("  that scores well on its test split and badly here has learned the")
        print("  dataset, not the task.")
    print("=" * 70)


if __name__ == "__main__":
    main()
