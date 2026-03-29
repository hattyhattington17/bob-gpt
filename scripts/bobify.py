"""One-time preprocessing: replace Shakespeare character names with Bob1, Bob2, etc.

Usage:
    uv run python scripts/bobify.py --input data/raw/shakespeare.txt

Download raw Shakespeare text from Project Gutenberg:
    curl -o data/raw/shakespeare.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt

Output is written to data/input.txt (gitignored).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Adjectives assigned to Bob aliases in order.
# Must have at least as many entries as SHAKESPEARE_NAMES.
BOB_ADJECTIVES: list[str] = [
    "Adventurous", "Ancient", "Angry", "Anxious", "Arcane",
    "Bashful", "Beefy", "Beloved", "Bewildered", "Blobby",
    "Bouncy", "Brainy", "Brave", "Breezy", "Bright",
    "Bumbling", "Cheerful", "Chilly", "Chubby", "Clumsy",
    "Crafty", "Cranky", "Crusty", "Curious", "Dainty",
    "Dapper", "Dazzling", "Dim", "Dizzy", "Droopy",
    "Dusty", "Eager", "Earnest", "Elegant", "Enormous",
    "Fancy", "Feisty", "Fierce", "Fluffy", "Foolish",
    "Freckled", "Friendly", "Frosty", "Fuzzy", "Gentle",
    "Giddy", "Gigantic", "Gloomy", "Glorious", "Glowing",
    "Grumpy", "Handsome", "Happy", "Hardy", "Haunted",
    "Hefty", "Heroic", "Honest", "Hopeful", "Hungry",
    "Icy", "Jolly", "Jovial", "Joyful", "Jumpy",
    "Lanky", "Lazy", "Lively", "Lonely", "Loud",
    "Lovely", "Lucky", "Lumpy", "Majestic", "Mellow",
    "Messy", "Mighty", "Misty", "Moody", "Mysterious",
    "Nervous", "Nimble", "Noble", "Noisy", "Odd",
    "Ornery", "Patient", "Peaceful", "Peculiar", "Perky",
    "Plump", "Pompous", "Prickly", "Proud", "Pudgy",
    "Puffy", "Puzzled", "Quirky", "Radiant", "Reckless",
    "Restless", "Rosy", "Rowdy", "Royal", "Rugged",
    "Rumpled", "Rusty", "Sassy", "Scholarly", "Scruffy",
    "Serene", "Shaggy", "Sharp", "Shy", "Silly",
    "Sleepy", "Slippery", "Sly", "Smug", "Sneaky",
    "Snobby", "Soggy", "Solemn", "Speedy", "Spiky",
    "Spirited", "Splendid", "Spooky", "Sprightly", "Squiggly",
    "Stately", "Stout", "Stubborn", "Sturdy", "Sunny",
    "Suspicious", "Swanky", "Swift", "Tall", "Tangled",
    "Tattered", "Tender", "Tiny", "Tired", "Touchy",
    "Tough", "Tricky", "Trusty", "Twitchy", "Unpredictable",
    "Valiant", "Vexed", "Vibrant", "Wandering", "Wary",
    "Whimsical", "Wicked", "Wiggly", "Wild", "Wise",
    "Witty", "Wobbly", "Woolly", "Wry", "Zany",
]

# All Shakespeare character names to replace.
# Sorted alphabetically so each name always maps to the same Bob alias.
SHAKESPEARE_NAMES: list[str] = sorted([
    "ADRIANA", "AEGEON", "AEMILIA", "AGAMEMNON", "AGRIPPA", "AJAX", "ALCIBIADES",
    "ALONSO", "ANGELO", "ANTONIO", "ARIEL", "ARMADO", "AUFIDIUS",
    "BALTHASAR", "BANQUO", "BAPTISTA", "BASSANIO", "BEATRICE", "BENEDICK",
    "BENVOLIO", "BIANCA", "BIONDELLO", "BOLINGBROKE", "BOTTOM", "BRUTUS",
    "CALIBAN", "CALPHURNIA", "CAPULET", "CASCA", "CASSIO", "CASSIUS",
    "CELIA", "CERIMON", "CESARIO", "CLAUDIUS", "CLEOPATRA", "CORDELIA",
    "CORNELIUS", "COSTARD", "CRESSIDA", "CYMBELINE",
    "DEMETRIUS", "DESDEMONA", "DIANA", "DOGBERRY", "DONALBAIN", "DUKE",
    "DUNCAN", "EDGAR", "EDMUND", "EMILIA", "ENOBARBUS",
    "FALSTAFF", "FESTE", "FLUTE", "FORD", "FORTINBRAS", "FRIAR",
    "GERTRUDE", "GONZALO", "GONERIL", "GRATIANO", "GREMIO",
    "HAMLET", "HECATE", "HELENA", "HENRY", "HERMIA", "HERMIONE",
    "HIPPOLYTA", "HORATIO", "HOTSPUR", "HYMEN",
    "IAGO", "IMOGEN", "IRIS",
    "JACQUENETTA", "JAQUES", "JESSICA", "JULIA", "JULIET", "JULIUS",
    "KATE", "KENT",
    "LAERTES", "LAUNCE", "LAUNCELOT", "LEAR", "LENNOX", "LEONATO",
    "LEONTES", "LEPIDUS", "LODOVICO", "LONGAVILLE", "LUCENTIO", "LUCIUS",
    "LYSANDER",
    "MACBETH", "MACDUFF", "MALCOLM", "MALVOLIO", "MARIANA", "MARINA",
    "MERCUTIO", "MIRANDA", "MONTAGUE", "MOTH",
    "NATHANIEL", "NERISSA", "OBERON", "OCTAVIA", "OCTAVIUS",
    "OLIVIA", "OPHELIA", "ORLANDO", "ORSINO", "OTHELLO",
    "PAGE", "PARIS", "PAULINA", "PERDITA", "PERICLES", "PETRUCHIO",
    "PHILOSTRATE", "PHRYNIA", "PISTOL", "POLONIUS", "PORTIA",
    "PROSPERO", "PROTEUS", "PUCK",
    "QUICKLY", "QUINCE",
    "REGAN", "RICHARD", "ROMEO", "ROSALIND", "ROSALINE", "ROSS",
    "SEBASTIAN", "SHYLOCK", "SILVIA", "SIMONIDES", "SNOUT", "SNUG",
    "STEPHANO", "STARVELING",
    "TAMING", "TIMON", "TITUS", "TITANIA", "TRANIO", "TRINCULO",
    "TROILUS", "TYBALT",
    "ULYSSES", "VALENTINE", "VINCENTIO", "VIOLA", "VIRGILIA",
    "WILLIAM",
])


def build_name_mapping(names: list[str], adjectives: list[str]) -> dict[str, str]:
    """Map each name to an adjective Bob alias (e.g. "Fluffy Bob").

    Args:
        names: Already-sorted list of character names.
        adjectives: List of adjectives, one per name in order.

    Returns:
        Dict mapping each name to its Bob alias.
    """
    assert len(adjectives) >= len(names), (
        f"Not enough adjectives ({len(adjectives)}) for names ({len(names)})"
    )
    return {name: f"{adjectives[i]} Bob" for i, name in enumerate(names)}


def strip_copyright_blocks(text: str) -> str:
    """Remove <<...>> copyright notice blocks from the text.

    Args:
        text: Raw input text.

    Returns:
        Text with all <<...>> blocks removed.
    """
    return re.sub(r"<<.*?>>", "", text, flags=re.DOTALL)


def bobify(text: str, mapping: dict[str, str]) -> str:
    """Replace all character name occurrences in text.

    Args:
        text: Raw input text.
        mapping: Dict of name → Bob alias.

    Returns:
        Text with all names replaced.
    """
    for name, alias in mapping.items():
        text = text.replace(name, alias)
    return text


def main() -> None:
    """Run the bobify preprocessing script."""
    parser = argparse.ArgumentParser(description="Replace Shakespeare names with Bob aliases.")
    parser.add_argument(
        "--input",
        default="data/raw/shakespeare.txt",
        help="Path to raw Shakespeare text (default: data/raw/shakespeare.txt)",
    )
    parser.add_argument(
        "--output",
        default="data/input.txt",
        help="Path to write bobified text (default: data/input.txt)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Download with:\n"
            "  mkdir -p data/raw\n"
            "  curl -o data/raw/shakespeare.txt "
            "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        )

    text = input_path.read_text(encoding="utf-8", errors="replace")
    text = strip_copyright_blocks(text)
    mapping = build_name_mapping(SHAKESPEARE_NAMES, BOB_ADJECTIVES)
    output = bobify(text, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

    vocab_size = len(set(output))
    print(f"Wrote {len(output):,} chars to {output_path}")
    print(f"Vocab size: {vocab_size} unique characters")
    print(f"Update nano.yaml: vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
