"""Deceptive and trap-laden reasoning benchmark suite.

Targets cognitive illusions, arithmetic distractors (Apple GSM-NoOp),
reasoning rigidity / instruction overriding (KAIST/AITRICS ReasoningTrap),
and counterfactual traps where standard Looped LM / CoT pattern-matching defaults
to a familiar incorrect answer (S0 trap), testing if latent excursion-and-return (E2)
re-anchors the model to the conditioned constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tebdil.tasks.literature_reasoning import extract_literature_integer


def build_deceptive_reasoning_dataset() -> Dict[str, Any]:
    """Construct the 38-task deceptive and hard reasoning benchmark."""
    suffix = "Be concise. Output only the integer result.\nAnswer: "

    tasks = [
        # --- Family 1: reasoning_rigidity_puzzles (ReasoningTrap / Counterfactuals) ---
        {
            "problem_id": "trap_infertile_rabbits",
            "family": "reasoning_rigidity",
            "template_id": "trap_rabbit_infertility_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Fibonacci Inversion",
            "reasoning_pathology": "habitual_fibonacci_overriding",
            "prompt": (
                "A pair of permanently infertile adult rabbits is placed in an enclosure on January 1st. "
                "Every month, each fertile mature pair produces one new pair of offspring. "
                "Rabbits take one month to mature. If no rabbits die or leave, how many total pairs "
                f"of rabbits will be in the enclosure on June 1st (after 5 months)? {suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["8", "5", "13", "6"],
        },
        {
            "problem_id": "trap_simple_jug_measure",
            "family": "reasoning_rigidity",
            "template_id": "trap_direct_jug_capacity_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Water Jug Inversion",
            "reasoning_pathology": "die_hard_pouring_overcomplication",
            "prompt": (
                "A gardener has four empty water jugs: a 2-gallon jug, a 3-gallon jug, a 4-gallon jug, "
                "and a 5-gallon jug. None of the jugs have intermediate measurement markings. "
                "To obtain exactly 3 gallons of water, what is the capacity in gallons of the single "
                f"jug that can be filled directly to the brim? {suffix}"
            ),
            "gold_answer": "3",
            "trap_answers": ["5", "4", "2", "6"],
        },
        {
            "problem_id": "trap_hotel_empty_rooms",
            "family": "reasoning_rigidity",
            "template_id": "trap_hilbert_unoccupied_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Hilbert Hotel Inversion",
            "reasoning_pathology": "unnecessary_hilbert_shift",
            "prompt": (
                "A grand hotel has an infinite number of rooms, but only 10 rooms are currently occupied "
                "and all other rooms are completely empty. A new guest arrives. "
                "What is the minimum number of currently registered guests that must be relocated "
                f"to another room in order to accommodate the new guest? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["10", "1", "11"],
        },
        {
            "problem_id": "trap_base10_reversal",
            "family": "reasoning_rigidity",
            "template_id": "trap_aime_base10_cycle_v1",
            "difficulty": "hard",
            "source_benchmark": "AIME 2022-I-2 Base Inversion",
            "reasoning_pathology": "memorized_aime_base9_recall",
            "prompt": (
                "Find the smallest three-digit positive integer abc (where a, b, and c are decimal digits "
                "with a > 0) such that the three-digit integer abc equals the three-digit integer bca "
                f"in base ten. {suffix}"
            ),
            "gold_answer": "111",
            "trap_answers": ["227", "100", "101"],
        },
        {
            "problem_id": "trap_potted_tree_soil",
            "family": "reasoning_rigidity",
            "template_id": "trap_van_helmont_tree_soil_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (PuzzleTrivial puzzle_32)",
            "reasoning_pathology": "naive_mass_subtraction_from_soil",
            "prompt": (
                "A 2 kg tree grows in a planted pot containing 10 kg of soil. Using water and carbon "
                "dioxide from the air via photosynthesis, the tree grows until its mass reaches 13 kg. "
                "Assuming negligible mineral mass loss from the soil, how many kilograms of soil "
                f"remain in the pot? {suffix}"
            ),
            "gold_answer": "10",
            "trap_answers": ["0", "1", "8", "11"],
        },
        {
            "problem_id": "trap_non_hanoi_disk_moves",
            "family": "reasoning_rigidity",
            "template_id": "trap_unrestricted_disk_transfer_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (PuzzleTrivial puzzle_62)",
            "reasoning_pathology": "memorized_tower_of_hanoi_2n_minus_1",
            "prompt": (
                "A monk must move a stack of 3 sacred disks from Altar A to Altar B, moving one disk "
                "at a time. Unlike the Tower of Hanoi, there is no rule restricting larger disks from "
                "being placed on top of smaller disks, and disks can be moved directly from Altar A "
                "to Altar B without using an intermediate peg. What is the minimum number of moves "
                f"required to transfer all 3 disks to Altar B? {suffix}"
            ),
            "gold_answer": "3",
            "trap_answers": ["7", "6", "8", "5"],
        },
        {
            "problem_id": "trap_identical_coins_weighing",
            "family": "reasoning_rigidity",
            "template_id": "trap_twelve_identical_coins_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (PuzzleTrivial puzzle_42)",
            "reasoning_pathology": "habitual_12_coin_balance_tree",
            "prompt": (
                "There are 12 coins on a table alongside a balance scale. Although a famous riddle "
                "claims one coin is heavier, in reality all 12 coins are genuine and have the exact "
                "same weight. How many heavier counterfeit coins are among the 12 coins? "
                f"{suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["3", "1", "2", "4"],
        },
        {
            "problem_id": "trap_aime_empty_intersection_sum",
            "family": "reasoning_rigidity",
            "template_id": "trap_aime_2022_empty_subset_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (AIME 2022-I-12_ver1)",
            "reasoning_pathology": "combinatorial_summation_ignoring_empty_set",
            "prompt": (
                "For any finite set X, let |X| denote the number of elements in X. Define S_n as the "
                "sum of |A intersect B| over all ordered pairs (A, B) of subsets of {1, 2, ..., n} "
                f"such that |A| = |B| and A is the empty set. Find S_10. {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["245", "1", "10", "1024"],
        },
        {
            "problem_id": "trap_aime_alice_bob_nim_take_all",
            "family": "reasoning_rigidity",
            "template_id": "trap_aime_2024_nim_take_all_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (AIME 2024 68_ver3)",
            "reasoning_pathology": "modular_nim_game_analysis_ignoring_instant_win",
            "prompt": (
                "Alice and Bob play a token game with an initial stack of n tokens (where n >= 1). "
                "Alice always plays first. On each turn, a player may remove either 1 token or n tokens "
                "(the entire initial stack size). Whoever removes the last token wins. For how many "
                "positive integers n less than or equal to 2024 does Bob have a guaranteed winning "
                f"strategy? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["809", "404", "674", "1", "1012"],
        },
        {
            "problem_id": "trap_aime_empty_grid_chips",
            "family": "reasoning_rigidity",
            "template_id": "trap_aime_2024_zero_chips_grid_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (AIME 2024 89_ver3)",
            "reasoning_pathology": "memorized_902_grid_coloring_overriding_zero_chips",
            "prompt": (
                "A collection of 25 indistinguishable white chips and 25 indistinguishable black chips "
                "is available to place on a 5x5 grid such that each cell contains at most one chip, "
                "all chips in the same row and all chips in the same column have the same color, and "
                "the total number of chips placed on the grid is 0 (the grid must remain completely "
                "empty). How many valid chip configurations on the 5x5 grid satisfy these conditions? "
                f"{suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["902", "0", "25", "2"],
        },
        {
            "problem_id": "trap_math_gecko_coincident",
            "family": "reasoning_rigidity",
            "template_id": "trap_math500_coincident_gecko_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (MATH500 geometry/880_ver4)",
            "reasoning_pathology": "unfolding_3d_room_walls_for_zero_distance",
            "prompt": (
                "A gecko is in a rectangular room that is 12 feet long, 10 feet wide, and 8 feet tall. "
                "Both the gecko and a fly are currently at the exact same point on the back wall, "
                "1 foot from the ceiling and 1 foot from the left wall. What is the length in feet "
                f"of the shortest path for the gecko to reach the fly? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["21", "22", "2", "18"],
        },
        {
            "problem_id": "trap_math_parentheses_addition",
            "family": "reasoning_rigidity",
            "template_id": "trap_math500_associative_parentheses_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (MATH500 prealgebra/1139_ver3)",
            "reasoning_pathology": "catalan_parenthesization_for_associative_sum",
            "prompt": (
                "How many distinct numerical values can be obtained from the arithmetic expression "
                "2 + 3 + 4 + 5 + 1 by inserting parentheses in different valid ways without "
                f"rearranging the numbers or changing the addition operators? {suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["14", "15", "4", "5"],
        },
        {
            "problem_id": "trap_math_log_single_solution",
            "family": "reasoning_rigidity",
            "template_id": "trap_math500_log_positive_k_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap (MATH500 intermediate_algebra/1849)",
            "reasoning_pathology": "memorized_501_including_negative_k_or_returning_k_value",
            "prompt": (
                "Find the number of positive integer values of k in the interval [1, 500] for which "
                "the real equation log_10(k * x) = 2 * log_10(x + 2) has exactly one real solution x. "
                f"{suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["501", "500", "8", "2", "0"],
        },
        # --- Family 2: gsm_noop_distractors (Apple AIML GSM-NoOp & BBEH Distractors) ---
        {
            "problem_id": "trap_gsm_kiwi_size",
            "family": "gsm_noop_distractor",
            "template_id": "noop_kiwi_under_average_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp (Mirzadeh et al., 2024)",
            "reasoning_pathology": "distractor_subtraction_bias",
            "prompt": (
                "Oliver picks 44 kiwis on Friday and 58 kiwis on Saturday. On Sunday, he picks double "
                "the number of kiwis he picked on Friday. On Sunday, 5 of the kiwis he picked were "
                "a bit smaller than average. How many total kiwis did Oliver pick across all three "
                f"days? {suffix}"
            ),
            "gold_answer": "190",
            "trap_answers": ["185", "180", "195"],
        },
        {
            "problem_id": "trap_gsm_school_inflation",
            "family": "gsm_noop_distractor",
            "template_id": "noop_school_historical_inflation_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp (Mirzadeh et al., 2024)",
            "reasoning_pathology": "historical_rate_application",
            "prompt": (
                "Liam buys 20 notebooks that cost $5 each and 15 pens that cost $2 each today. "
                "Due to general economic inflation, prices of stationery were 10 percent lower last year. "
                f"How many dollars does Liam pay in total for his purchase today? {suffix}"
            ),
            "gold_answer": "130",
            "trap_answers": ["117", "143", "120"],
        },
        {
            "problem_id": "trap_gsm_apple_crates",
            "family": "gsm_noop_distractor",
            "template_id": "noop_crate_shelf_placement_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp / Inventory Placement",
            "reasoning_pathology": "spatial_subset_exclusion",
            "prompt": (
                "A warehouse stores 15 crates of apples, and each crate contains exactly 20 apples. "
                "During routine morning restocking, 3 of the crates were placed on the top shelf. "
                f"How many total apples are stored in the warehouse? {suffix}"
            ),
            "gold_answer": "300",
            "trap_answers": ["240", "60", "260"],
        },
        {
            "problem_id": "trap_gsm_bakery_shapes",
            "family": "gsm_noop_distractor",
            "template_id": "noop_bakery_cookie_shape_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp / Attribute Irrelevance",
            "reasoning_pathology": "shape_filter_deduction",
            "prompt": (
                "A baker bakes 60 chocolate cookies and 40 butter cookies. Of all the cookies baked, "
                "15 cookies were shaped like stars and the rest were shaped like circles. "
                f"How many total cookies did the baker bake? {suffix}"
            ),
            "gold_answer": "100",
            "trap_answers": ["85", "75", "115"],
        },
        {
            "problem_id": "trap_gsm_noop_andre_raise",
            "family": "gsm_noop_distractor",
            "template_id": "noop_andre_september_wage_v1",
            "difficulty": "hard",
            "source_benchmark": "Experimental-Orange/gsm-noop-audited (ID 60)",
            "reasoning_pathology": "prior_month_wage_distractor",
            "prompt": (
                "Andre works a 12-hour shift each day, 6 days a week. He earns $13 per hour in October, "
                "having received a raise from $11 per hour back in September. He also gets a $132 bonus "
                "each week if the company performs well. How many dollars did Andre make in 4 weeks "
                f"of October if the company performed well every week? {suffix}"
            ),
            "gold_answer": "4272",
            "trap_answers": ["3696", "3744", "4140", "3168"],
        },
        {
            "problem_id": "trap_gsm_noop_phone_booth_weekend",
            "family": "gsm_noop_distractor",
            "template_id": "noop_phone_booth_weekend_rate_v1",
            "difficulty": "hard",
            "source_benchmark": "Experimental-Orange/gsm-noop-audited (ID 10)",
            "reasoning_pathology": "irrelevant_weekend_tier_substitution",
            "prompt": (
                "To make a weekday call from a public phone booth, you must pay $6 per minute for the "
                "first 30 minutes, and $5 per minute for every minute after the first 30 minutes. "
                "On weekends, the initial rate for the first 30 minutes drops to $4 per minute. "
                f"How many dollars does a 78-minute weekday call cost in total? {suffix}"
            ),
            "gold_answer": "420",
            "trap_answers": ["360", "468", "390", "312"],
        },
        {
            "problem_id": "trap_gsm_noop_kai_car_rental",
            "family": "gsm_noop_distractor",
            "template_id": "noop_kai_late_fee_v1",
            "difficulty": "hard",
            "source_benchmark": "Experimental-Orange/gsm-noop-audited (ID 89)",
            "reasoning_pathology": "untriggered_late_fee_addition",
            "prompt": (
                "Kai rents a car from 1 PM to 5 PM on the same day (4 hours in total). He gets the "
                "first 2 hours free. The first paid hour costs $10, and each subsequent paid hour "
                "costs twice as much as the immediately preceding paid hour. The rental company also "
                "charges a $15 late fee if the car is returned after 6 PM. How many dollars does Kai "
                f"pay in total when he returns the car at 5 PM? {suffix}"
            ),
            "gold_answer": "30",
            "trap_answers": ["45", "70", "150", "85"],
        },
        {
            "problem_id": "trap_bbeh_retroactive_inventory",
            "family": "gsm_noop_distractor",
            "template_id": "bbeh_object_counting_retroactive_v1",
            "difficulty": "hard",
            "source_benchmark": "Google DeepMind BBEH (bbeh_object_counting)",
            "reasoning_pathology": "non_target_entity_inclusion_and_retroactive_drift",
            "prompt": (
                "A collector is taking inventory of only the musical instruments in a studio. "
                "The studio contains 4 pianos, 5 violins, 12 wooden chairs, 3 flutes, 1 trombone, "
                "8 oil paintings, and a collection of accordions (the studio initially had 56 accordions, "
                "then sold 46 of them, then bought 52 more, and finally donated 40 of them). "
                f"How many musical instruments in total are currently in the studio? {suffix}"
            ),
            "gold_answer": "35",
            "trap_answers": ["55", "25", "39", "14"],
        },
        # --- Family 3: cognitive_illusions (CRT, SimpleBench & XERJ MT-Bench Riddles) ---
        {
            "problem_id": "trap_speed_harmonic_mean",
            "family": "cognitive_illusion",
            "template_id": "trap_roundtrip_harmonic_speed_v1",
            "difficulty": "medium",
            "source_benchmark": "Classic Physics / SimpleBench Cognitive Traps",
            "reasoning_pathology": "arithmetic_mean_heuristic",
            "prompt": (
                "A delivery van travels from Warehouse A to Warehouse B at an average speed of 30 km/h, "
                "and immediately returns along the exact same road from Warehouse B to Warehouse A "
                "at an average speed of 60 km/h. What is the average speed of the van for the entire "
                f"round trip in km/h? {suffix}"
            ),
            "gold_answer": "40",
            "trap_answers": ["45", "50", "90"],
        },
        {
            "problem_id": "trap_bat_ball_glove",
            "family": "cognitive_illusion",
            "template_id": "trap_three_item_shifted_crt_v1",
            "difficulty": "medium",
            "source_benchmark": "Modified CRT / Frederick Cognitive Reflection Test",
            "reasoning_pathology": "superficial_subtraction_heuristic",
            "prompt": (
                "A baseball bat, a ball, and a leather glove cost $150 in total. The glove costs $40. "
                "The bat costs $100 more than the ball. How many dollars does the ball cost? "
                f"{suffix}"
            ),
            "gold_answer": "5",
            "trap_answers": ["10", "25", "50"],
        },
        {
            "problem_id": "trap_lily_pad_halving",
            "family": "cognitive_illusion",
            "template_id": "trap_exponential_halving_day_v1",
            "difficulty": "medium",
            "source_benchmark": "Cognitive Reflection Test / Doubling Rate",
            "reasoning_pathology": "linear_halving_heuristic",
            "prompt": (
                "A patch of water lilies in a lake doubles in size every single day. If it takes exactly "
                "48 days for the patch to completely cover the entire surface of the lake, "
                f"on what day was the lake exactly half covered? {suffix}"
            ),
            "gold_answer": "47",
            "trap_answers": ["24", "46", "12"],
        },
        {
            "problem_id": "trap_monty_hall_4doors",
            "family": "cognitive_illusion",
            "template_id": "trap_monty_hall_4door_switch_v1",
            "difficulty": "hard",
            "source_benchmark": "4-Door Monty Hall Probability Problem",
            "reasoning_pathology": "equal_split_fallacy",
            "prompt": (
                "In a game show, there are 4 closed doors. Behind 1 door is a luxury car, and behind "
                "each of the other 3 doors is a goat. You initially choose Door 1. "
                "The host, who knows where the car is, opens Door 2 and Door 3, revealing goats behind both. "
                "The host then offers you the option to switch to Door 4. "
                "What is the probability (expressed as an integer percentage from 0 to 100) of winning "
                f"the car if you switch to Door 4? {suffix}"
            ),
            "gold_answer": "75",
            "trap_answers": ["50", "25", "33"],
        },
        {
            "problem_id": "trap_simplebench_frying_pan_ice",
            "family": "cognitive_illusion",
            "template_id": "trap_simplebench_melting_ice_v1",
            "difficulty": "hard",
            "source_benchmark": "SimpleBench Public (Q1)",
            "reasoning_pathology": "blind_arithmetic_summation_ignoring_phase_change",
            "prompt": (
                "Beth places four whole ice cubes in a hot frying pan at the start of the first minute, "
                "then five whole ice cubes at the start of the second minute, and six more whole ice "
                "cubes at the start of the third minute while frying a crispy egg in the pan. "
                "If the pan remains at egg-frying temperature throughout, what is the number of whole "
                f"solid ice cubes left in the pan at the end of the third minute? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["15", "5", "6", "30"],
        },
        {
            "problem_id": "trap_simplebench_fruit_shelf_eaten",
            "family": "cognitive_illusion",
            "template_id": "trap_simplebench_color_scarf_fruits_v1",
            "difficulty": "hard",
            "source_benchmark": "SimpleBench Public (Q8)",
            "reasoning_pathology": "color_matching_assumption_ignoring_total_depletion",
            "prompt": (
                "On a shelf, there are five fruits in total: a green apple, a red pear, a pink peach, "
                "a yellow banana, and a purple plum, arranged in that order. Three boys walk into the "
                "room wearing a green scarf, a red scarf, and a pink scarf, respectively. The red-scarfed "
                "boy eats the red pear, and the green-scarfed boy eats the green apple and three other "
                "fruits from the shelf. How many fruits remain on the shelf for the pink-scarfed boy "
                f"to eat? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["1", "2", "3", "4"],
        },
        {
            "problem_id": "trap_mtbench_david_sisters_brothers",
            "family": "cognitive_illusion",
            "template_id": "trap_mtbench_self_brother_inclusion_v1",
            "difficulty": "hard",
            "source_benchmark": "XERJ Corpus Hub: fastchat (pin 2026-10-13, Apache-2.0, FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl:24)",
            "reasoning_pathology": "egocentric_sibling_count_exclusion",
            "prompt": (
                "David (who is male) has three sisters. Each of his sisters has exactly two brothers. "
                f"How many brothers does David have? {suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["2", "6", "3", "0"],
        },
        # --- Family 4: rate_and_combinatorics (XERJ MT-Bench, MATH-500 L5, & Combinatoric Traps) ---
        {
            "problem_id": "trap_widget_machines",
            "family": "rate_and_combinatorics",
            "template_id": "trap_unit_rate_scaling_v1",
            "difficulty": "medium",
            "source_benchmark": "Cognitive Reflection Test / Machine Scaling",
            "reasoning_pathology": "superficial_proportionality_scaling",
            "prompt": (
                "If 5 identical automated machines take 5 minutes to manufacture 5 widgets, "
                "how many minutes does it take 100 identical automated machines to manufacture "
                f"100 widgets? {suffix}"
            ),
            "gold_answer": "5",
            "trap_answers": ["100", "20", "1"],
        },
        {
            "problem_id": "trap_clock_strikes",
            "family": "rate_and_combinatorics",
            "template_id": "trap_fencepost_interval_chimes_v1",
            "difficulty": "hard",
            "source_benchmark": "Classic Arithmetic Fencepost Riddle / SimpleBench",
            "reasoning_pathology": "fencepost_interval_neglect",
            "prompt": (
                "An antique tower clock takes 6 seconds to complete striking 4 o'clock (4 chimes separated "
                "by equal silent intervals). How many seconds does the clock take to complete striking "
                f"10 o'clock (10 chimes) at the exact same chiming rate? {suffix}"
            ),
            "gold_answer": "18",
            "trap_answers": ["15", "20", "12"],
        },
        {
            "problem_id": "trap_couple_handshakes",
            "family": "rate_and_combinatorics",
            "template_id": "trap_spouse_exclusion_handshakes_v1",
            "difficulty": "hard",
            "source_benchmark": "Discrete Combinatorics / SimpleBench Social Traps",
            "reasoning_pathology": "spouse_exclusion_neglect",
            "prompt": (
                "Three married couples (6 people in total) attend a dinner party. Each person shakes hands "
                "with every other person in the room exactly once, except that no one shakes hands "
                f"with their own spouse. How many total handshakes occur? {suffix}"
            ),
            "gold_answer": "12",
            "trap_answers": ["15", "9", "18"],
        },
        {
            "problem_id": "trap_rope_burn_timing",
            "family": "rate_and_combinatorics",
            "template_id": "trap_nonuniform_rope_burning_v1",
            "difficulty": "hard",
            "source_benchmark": "Non-uniform Rope Burning Puzzle",
            "reasoning_pathology": "halving_interpolation_error",
            "prompt": (
                "You have two identical ropes. Each rope burns completely in 60 minutes from end to end, "
                "but the ropes burn non-uniformly (different sections burn at different speeds). "
                "By lighting Rope 1 at both ends and Rope 2 at one end simultaneously, and lighting the "
                "second end of Rope 2 the moment Rope 1 finishes burning, you measure an exact duration. "
                f"How many total minutes of time elapse from the start until Rope 2 burns out completely? {suffix}"
            ),
            "gold_answer": "45",
            "trap_answers": ["60", "30", "15"],
        },
        {
            "problem_id": "trap_mtbench_bus_reverse_flow",
            "family": "rate_and_combinatorics",
            "template_id": "trap_mtbench_bus_passengers_reverse_v1",
            "difficulty": "hard",
            "source_benchmark": "XERJ Corpus Hub: fastchat (pin 2026-10-13, Apache-2.0, FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl:35)",
            "reasoning_pathology": "forward_inverse_operation_order_error",
            "prompt": (
                "When a commuter bus leaves the terminal, a certain number of passengers are on board. "
                "At the first bus stop, half of the passengers get off and 4 new passengers get on. "
                "At the second bus stop, 6 passengers get off and 8 new passengers get on. "
                "If there are 25 passengers on the bus heading toward the third stop, how many "
                f"passengers were on the bus when it originally left the terminal? {suffix}"
            ),
            "gold_answer": "38",
            "trap_answers": ["42", "34", "50", "19"],
        },
        {
            "problem_id": "trap_mtbench_survey_neither",
            "family": "rate_and_combinatorics",
            "template_id": "trap_mtbench_inclusion_exclusion_v1",
            "difficulty": "medium",
            "source_benchmark": "XERJ Corpus Hub: fastchat (pin 2026-10-13, Apache-2.0, FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl:33)",
            "reasoning_pathology": "double_counting_intersection_neglect",
            "prompt": (
                "In a survey conducted at a high school, 58% of students like the color blue, "
                "45% of students like the color green, and 22% of students like both blue and green. "
                "Express as an integer percentage (from 0 to 100) the probability that a randomly "
                f"chosen student likes neither blue nor green. {suffix}"
            ),
            "gold_answer": "19",
            "trap_answers": ["3", "22", "21", "81"],
        },
        {
            "problem_id": "trap_mtbench_executive_parking_zebra",
            "family": "rate_and_combinatorics",
            "template_id": "trap_mtbench_5exec_parking_zebra_v1",
            "difficulty": "hard",
            "source_benchmark": "XERJ Corpus Hub: fastchat (pin 2026-10-13, Apache-2.0, FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl:25)",
            "reasoning_pathology": "coupled_color_and_person_adjacency_inversion",
            "prompt": (
                "Five executives — Alice (1), Bert (2), Cheryl (3), David (4), and Enid (5) — park "
                "their cars in a row of 5 reserved spaces numbered 1 to 5 from left to right.\n"
                "- The car in Space 1 is red, the car in Space 5 is purple, and the blue car is parked "
                "immediately between the red car and the green car.\n"
                "- The remaining car is yellow.\n"
                "- Enid drives the green car.\n"
                "- Bert's car is parked immediately between Cheryl's car and Enid's car.\n"
                "- David's car is parked in Space 5, and Alice's car is parked immediately next to David's car.\n"
                "Representing Alice=1, Bert=2, Cheryl=3, David=4, and Enid=5, what is the 5-digit "
                f"integer formed by the executives' numbers from Space 1 to Space 5? {suffix}"
            ),
            "gold_answer": "32514",
            "trap_answers": ["12534", "32541", "41523", "32154"],
        },
        {
            "problem_id": "trap_math500_amicable_284",
            "family": "rate_and_combinatorics",
            "template_id": "trap_math500_amicable_divisors_v1",
            "difficulty": "hard",
            "source_benchmark": "HuggingFaceH4/MATH-500 (test/number_theory/737.json, Level 5)",
            "reasoning_pathology": "single_iteration_divisor_sum_or_including_N",
            "prompt": (
                "A proper divisor of a positive integer N is a positive divisor of N that is strictly "
                "less than N. What is the sum of the proper divisors of the sum of the proper divisors "
                f"of 284? {suffix}"
            ),
            "gold_answer": "284",
            "trap_answers": ["220", "504", "283", "142"],
        },
        {
            "problem_id": "trap_math500_round_table_144",
            "family": "rate_and_combinatorics",
            "template_id": "trap_math500_circular_nonadjacent_v1",
            "difficulty": "hard",
            "source_benchmark": "HuggingFaceH4/MATH-500 (test/counting_and_probability/525.json, Level 5)",
            "reasoning_pathology": "linear_gap_count_or_missing_rotational_symmetry",
            "prompt": (
                "In how many ways can 7 people sit around a circular table if no two of the 3 people "
                "Pierre, Rosa, and Thomas can sit immediately next to each other? Two seating "
                f"arrangements are considered identical if one is a rotation of the other. {suffix}"
            ),
            "gold_answer": "144",
            "trap_answers": ["720", "288", "576", "36"],
        },
        {
            "problem_id": "trap_math500_monic_factors_10",
            "family": "rate_and_combinatorics",
            "template_id": "trap_math500_sophie_germain_factors_v1",
            "difficulty": "hard",
            "source_benchmark": "HuggingFaceH4/MATH-500 (test/intermediate_algebra/279.json, Level 5)",
            "reasoning_pathology": "missing_sophie_germain_x4_plus_4_factorization",
            "prompt": (
                "Suppose that the polynomial x^8 + 3x^4 - 4 is factored as p_1(x) * p_2(x) * ... * p_k(x), "
                "where each p_i(x) is a non-constant monic polynomial with integer coefficients that "
                "cannot be factored further over the integers (irreducible over Z). Compute the sum "
                f"p_1(1) + p_2(1) + ... + p_k(1). {suffix}"
            ),
            "gold_answer": "10",
            "trap_answers": ["9", "7", "0", "8"],
        },
    ]

    _verify_tasks(tasks)
    return {
        "schema_version": 1,
        "suite": "deceptive_traps_v1",
        "purpose": (
            "Deceptive and hard reasoning benchmark targeting cognitive illusions, arithmetic distractors (GSM-NoOp/BBEH), "
            "reasoning rigidity (ReasoningTrap), and combinatoric/algebraic traps (XERJ MT-Bench, MATH-500 L5) where S0 habitually outputs trap answers."
        ),
        "families": [
            "reasoning_rigidity",
            "gsm_noop_distractor",
            "cognitive_illusion",
            "rate_and_combinatorics",
        ],
        "tasks": tasks,
    }


def _verify_tasks(tasks: List[Dict[str, Any]]) -> None:
    seen_ids = set()
    by_id = {}
    for t in tasks:
        tid = t["problem_id"]
        assert tid not in seen_ids, f"Duplicate problem_id: {tid}"
        seen_ids.add(tid)
        by_id[tid] = t
        gold = t["gold_answer"]
        assert isinstance(gold, str) and gold.strip(), f"Invalid gold_answer for {tid}"
        int(gold)  # must be integer
        traps = t.get("trap_answers", [])
        assert isinstance(traps, list) and len(traps) >= 1, f"Missing trap_answers for {tid}"
        for trap in traps:
            assert trap != gold, f"Trap answer '{trap}' cannot equal gold '{gold}' in {tid}"

    # Programmatic verification of mathematical / combinatorial gold answers
    assert int(by_id["trap_math_log_single_solution"]["gold_answer"]) == sum(
        1 for k in range(1, 501) if (4 - k) ** 2 - 16 == 0 and k - 4 > 0
    ) == 1
    assert int(by_id["trap_gsm_noop_andre_raise"]["gold_answer"]) == 12 * 13 * 6 * 4 + 132 * 4 == 4272
    assert int(by_id["trap_gsm_noop_phone_booth_weekend"]["gold_answer"]) == 30 * 6 + (78 - 30) * 5 == 420
    assert int(by_id["trap_gsm_noop_kai_car_rental"]["gold_answer"]) == 10 + 20 == 30
    assert int(by_id["trap_bbeh_retroactive_inventory"]["gold_answer"]) == 4 + 5 + 3 + 1 + (56 - 46 + 52 - 40) == 35
    assert int(by_id["trap_mtbench_bus_reverse_flow"]["gold_answer"]) == ((25 - 8 + 6) - 4) * 2 == 38
    assert int(by_id["trap_mtbench_survey_neither"]["gold_answer"]) == 100 - (58 + 45 - 22) == 19
    s284 = sum(d for d in range(1, 284) if 284 % d == 0)
    s220 = sum(d for d in range(1, s284) if s284 % d == 0)
    assert s284 == 220 and s220 == 284 and int(by_id["trap_math500_amicable_284"]["gold_answer"]) == 284
    assert int(by_id["trap_math500_round_table_144"]["gold_answer"]) == 24 * 6 == 144
    assert int(by_id["trap_math500_monic_factors_10"]["gold_answer"]) == (1 - 1) + (1 + 1) + (1 + 1) + (1 - 2 + 2) + (1 + 2 + 2) == 10


def evaluate_deceptive_reasoning(row: Dict[str, Any], gold_answer: str) -> Dict[str, Any]:
    """Evaluate a generated deceptive reasoning completion against gold and trap answers."""
    final_text = row.get("final_text")
    completed = bool(row.get("completed", False))
    extracted = extract_literature_integer(final_text, completed=completed)

    trap_answers = set(row.get("trap_answers", []))
    hit_trap = extracted in trap_answers if extracted is not None else False

    if not completed and extracted is None:
        return {
            "status": "budget_exhausted",
            "extracted_answer": None,
            "is_correct": False,
            "hit_trap_answer": False,
            "gold_answer": gold_answer,
        }

    if extracted is None:
        return {
            "status": "invalid_format",
            "extracted_answer": None,
            "is_correct": False,
            "hit_trap_answer": False,
            "gold_answer": gold_answer,
        }

    is_correct = str(extracted).strip() == str(gold_answer).strip()
    return {
        "status": "correct" if is_correct else ("trap_failure" if hit_trap else "incorrect"),
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "hit_trap_answer": hit_trap,
        "gold_answer": gold_answer,
    }


def write_deceptive_tasks_config(path: Path) -> Dict[str, Any]:
    data = build_deceptive_reasoning_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
