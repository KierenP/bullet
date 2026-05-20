use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use bullet_lib::{
    game::{
        formats::{bulletformat::ChessBoard, montyformat::chess::Attacks},
        inputs::{SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::optimiser::{Ranger, RangerParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{ViriBinpackLoader, viribinpack},
    },
};

//----------------------------------
// Threat Inputs Implementation
//----------------------------------

/// Threat inputs specification:
///
/// In a normal king-bucket NNUE, each input is a (piece, square, king-bucket) tuple. For threat inputs, we keep the
/// (piece, square, king-bucket) inputs but add an additional set of inputs which are (piece, square, threat-piece,
/// threat-square). I.e 'white bishop on c4 is attacking black knight on f7' would activate the input corresponding to
/// (white bishop, c4, black knight, f7).
///
/// This would result in a very large number of inputs (12 pieces * 64 squares * 64 threat-squares * 12 threat-pieces
/// = 589,824) which is too large to be practical. To reduce this, we can observe that not all threat inputs are
/// possible. Depending on the piece and square, only certain target squares can be attacked. For example, a knight on
/// c4 can only attack 8 squares, so it can only activate 8*12=96 threat inputs, not 64*12=768.
///
/// We further reduce the input count by restricting which piece types can threaten which, because some threats are
/// symmetric. E.g rook -> queen implies queen -> rook.
/// - Pawn only threatens pawns, knights, and rooks (6 victims)
/// - Knight threatens everyone (12 victims)
/// - Bishop/rook don't threaten queens (10 victims)
/// - Queen threatens everyone (12 victims)
/// - King only threatens pawns, knights, bishops, and rooks (8 victims)
///
/// - pawn:   84 attacks * 6 victims = 504
/// - knight: 336 attacks * 12 victims = 4,032
/// - bishop: 560 attacks * 10 victims = 5,600
/// - rook:   896 attacks * 10 victims = 8,960
/// - queen:  1456 attacks * 12 victims = 17,472
/// - king:   420 attacks * 8 victims  = 3,360
/// - TOTAL: (504 + 4032 + 5600 + 8960 + 17472 + 3360) * 2 attacker sides = 79,856 threat inputs
///
/// For speed and simplicity, we precompute lookup tables for each (piece, square) with the offset into the threat
/// input list. That way we can enumerate the attack mask and efficiently activate the relevant threat inputs without
/// needing to do any complex calculations at runtime.
///
/// The threat inputs are added alongside the usual (piece, square, king-bucket) inputs.

// ============================================================
// Attack generation
// ============================================================

/// Piece types in ChessBoard encoding: bits 0-2
const PAWN: u8 = 0;
const KNIGHT: u8 = 1;
const BISHOP: u8 = 2;
const ROOK: u8 = 3;
const QUEEN: u8 = 4;
const KING: u8 = 5;

/// Get attack bitboard for a given piece type on a given square.
/// `side` is 0 for STM, 1 for NSTM (only matters for pawns).
fn attacks_for(piece_type: u8, sq: usize, side: usize, occ: u64) -> u64 {
    match piece_type {
        PAWN => Attacks::pawn(sq, side),
        KNIGHT => Attacks::knight(sq),
        BISHOP => Attacks::bishop(sq, occ),
        ROOK => Attacks::rook(sq, occ),
        QUEEN => Attacks::queen(sq, occ),
        KING => Attacks::king(sq),
        _ => 0,
    }
}

// ============================================================
// Threat table construction
// ============================================================

/// Check if an attacker piece type is allowed to threaten a victim piece type.
fn can_threaten(atk_piece: u8, vic_piece: u8) -> bool {
    match atk_piece {
        PAWN => matches!(vic_piece, PAWN | KNIGHT | ROOK),
        BISHOP | ROOK => vic_piece != QUEEN,
        KING => matches!(vic_piece, PAWN | KNIGHT | BISHOP | ROOK),
        _ => true,
    }
}

/// Precomputed lookup table mapping (attacker, square, victim, square) -> feature index.
struct ThreatTables {
    /// Total number of threat features (per perspective).
    total_threat_features: usize,

    /// Direct lookup: [atk_idx][atk_sq][vic_idx][vic_sq] -> feature index.
    /// atk_idx = piece_type * 2 + side, vic_idx = piece_type * 2 + side.
    /// u32::MAX = invalid/excluded/deduped threat.
    lookup: Box<[[[[u32; 64]; 12]; 64]; 12]>,
}

impl ThreatTables {
    fn new() -> Self {
        let mut lookup = Box::new([[[[u32::MAX; 64]; 12]; 64]; 12]);

        let mut current_offset: u32 = 0;

        for atk_piece in 0u8..6 {
            for atk_side in 0..2usize {
                let atk_idx = atk_piece as usize * 2 + atk_side;

                for atk_sq in 0..64usize {
                    // Pawns can't be on rank 0 (back rank) or rank 7 (promotion rank)
                    if atk_piece == PAWN && (atk_sq / 8 == 0 || atk_sq / 8 == 7) {
                        continue;
                    }

                    let attack_bb = attacks_for(atk_piece, atk_sq, atk_side, 0);
                    if attack_bb == 0 {
                        continue;
                    }

                    for vic_piece in 0u8..6 {
                        for vic_side in 0..2usize {
                            let vic_idx = vic_piece as usize * 2 + vic_side;

                            if !can_threaten(atk_piece, vic_piece) {
                                continue;
                            }

                            let mut bb = attack_bb;
                            while bb != 0 {
                                let target_sq = bb.trailing_zeros() as usize;
                                bb &= bb - 1;

                                lookup[atk_idx][atk_sq][vic_idx][target_sq] = current_offset;
                                current_offset += 1;
                            }
                        }
                    }
                }
            }
        }

        let total_threat_features = current_offset as usize;

        assert_eq!(
            total_threat_features, 79856,
            "Threat table size mismatch! Expected 79856, got {total_threat_features}",
        );

        Self { total_threat_features, lookup }
    }

    /// Get the threat feature index for an attacker threatening a victim.
    /// Returns None if this threat is not tracked (duplicate or excluded).
    #[inline]
    fn threat_feature(
        &self,
        atk_piece: u8,
        atk_side: usize,
        atk_sq: usize,
        vic_piece: u8,
        vic_side: usize,
        vic_sq: usize,
    ) -> Option<usize> {
        let atk_idx = atk_piece as usize * 2 + atk_side;
        let vic_idx = vic_piece as usize * 2 + vic_side;

        let idx = self.lookup[atk_idx][atk_sq][vic_idx][vic_sq];
        if idx == u32::MAX {
            return None;
        }

        Some(idx as usize)
    }
}

static THREAT_TABLES: OnceLock<ThreatTables> = OnceLock::new();

fn get_threat_tables() -> &'static ThreatTables {
    THREAT_TABLES.get_or_init(ThreatTables::new)
}

// ============================================================
// ChessBucketsMirroredWithThreats
// ============================================================

/// Every piece always activates a king-bucketed (piece, square) feature.
/// Pieces with active threats additionally activate threat features.
///
/// Feature layout:
///   [0, 768 * num_buckets)               : king-bucketed piece-square
///   [768 * num_buckets, +768)            : manual factorizer for king-bucketed portion (see below)
///   [768 * num_buckets, ...)             : threat features
#[derive(Clone)]
struct ChessBucketsMirroredWithThreats {
    buckets: [usize; 64],
    num_buckets: usize,
    /// Base offset for unbucketed piece-square features (= 768 * num_buckets)
    factorizer_base: usize,
    /// Base offset for threat features (= 768 * num_buckets + 768)
    threat_base: usize,
    /// Total number of inputs
    total_inputs: usize,
    threat_tables: &'static ThreatTables,
}

impl ChessBucketsMirroredWithThreats {
    fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        let threat_tables = get_threat_tables();
        let factorizer_base = 768 * num_buckets;
        let threat_base = factorizer_base + 768;
        let total_inputs = threat_base + threat_tables.total_threat_features;

        Self { buckets: expanded, num_buckets, factorizer_base, threat_base, total_inputs, threat_tables }
    }
}

/// Extract piece list from a ChessBoard's packed representation.
/// Returns array of (piece_nibble, square) pairs and the count.
#[inline]
fn extract_pieces(pos: &ChessBoard) -> ([(u8, usize); 32], usize) {
    let mut result = [(0u8, 0usize); 32];
    let mut count = 0;
    let mut occ = pos.occ();
    while occ != 0 {
        let sq = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        let piece = (pos.pcs[count / 2] >> (4 * (count & 1))) & 0b1111;
        result[count] = (piece, sq);
        count += 1;
    }
    (result, count)
}

impl SparseInputType for ChessBucketsMirroredWithThreats {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        self.total_inputs
    }

    fn max_active(&self) -> usize {
        // 32 king-bucketed + 32 unbucketed + threat features per piece
        32 + 32 + 96
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let tables = self.threat_tables;

        // Determine king-side flips and bucket offsets (same as ChessBucketsMirrored)
        let our_ksq = pos.our_ksq() as usize;
        let opp_ksq = pos.opp_ksq() as usize;
        let stm_flip = if our_ksq % 8 > 3 { 7 } else { 0 };
        let ntm_flip = if opp_ksq % 8 > 3 { 7 } else { 0 };
        let stm_bucket = 768 * self.buckets[our_ksq];
        let ntm_bucket = 768 * self.buckets[opp_ksq];

        // Extract piece data
        let (pieces, count) = extract_pieces(pos);
        let occ = pos.occ();

        // Build a per-square lookup: piece_on[sq] = piece_nibble (0xFF = empty)
        let mut piece_on = [0xFFu8; 64];
        for i in 0..count {
            let (piece, sq) = pieces[i];
            piece_on[sq] = piece;
        }

        // For each piece, emit king-bucketed features and compute threats
        for i in 0..count {
            let (piece, sq) = pieces[i];
            let c = ((piece >> 3) & 1) as usize; // 0=STM, 1=NSTM
            let pc = 64 * (piece & 7) as usize;

            // King-bucketed feature (same as ChessBucketsMirrored)
            let stm_feat = [0, 384][c] + pc + sq;
            let ntm_feat = [384, 0][c] + pc + (sq ^ 56);
            f(stm_bucket + (stm_feat ^ stm_flip), ntm_bucket + (ntm_feat ^ ntm_flip));

            // Compute attack set for this piece
            let piece_type = piece & 7;
            let attack_bb = attacks_for(piece_type, sq, c, occ);

            // Factorized feature
            f(self.factorizer_base + (stm_feat ^ stm_flip), self.factorizer_base + (ntm_feat ^ ntm_flip));

            // Find all pieces this piece attacks and emit threat features
            let attacked_pieces = attack_bb & occ;
            if attacked_pieces != 0 {
                let mut att = attacked_pieces;
                while att != 0 {
                    let target_sq = att.trailing_zeros() as usize;
                    att &= att - 1;

                    let vic_nibble = piece_on[target_sq];
                    if vic_nibble == 0xFF {
                        continue;
                    }

                    let vic_pt = vic_nibble & 7;
                    let vic_side = ((vic_nibble >> 3) & 1) as usize;

                    // Look up threat feature for both perspectives.
                    let stm_idx = match tables.threat_feature(
                        piece_type,
                        c,
                        sq ^ stm_flip,
                        vic_pt,
                        vic_side,
                        target_sq ^ stm_flip,
                    ) {
                        Some(idx) => idx,
                        None => continue, // not a tracked threat (can_threaten restriction)
                    };

                    // NTM perspective: flip sides and mirror vertically
                    let ntm_idx = match tables.threat_feature(
                        piece_type,
                        c ^ 1,
                        (sq ^ 56) ^ ntm_flip,
                        vic_pt,
                        vic_side ^ 1,
                        (target_sq ^ 56) ^ ntm_flip,
                    ) {
                        Some(idx) => idx,
                        None => continue,
                    };

                    f(self.threat_base + stm_idx, self.threat_base + ntm_idx);
                }
            }
        }
    }

    fn shorthand(&self) -> String {
        let tables = get_threat_tables();
        format!("768x{}hm+768+{}t", self.num_buckets, tables.total_threat_features)
    }

    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt chess inputs with threat inputs".to_string()
    }
}

//----------------------------------
use viriformat::chess::board::Board;

/// Enable debug printing of piece count distribution statistics
const DEBUG_DISTRIBUTION: bool = false;
/// How often to print debug info (every N positions seen)
const DEBUG_PRINT_INTERVAL: u64 = 10_000_000;

/// Target distribution for piece counts [0..=32].
/// Values represent the desired relative frequency (will be normalized).
/// Higher values = more positions with that piece count in the output.
/// Values based on Stockfish's PyTorch NNUE trainer (0-2 pieces set to 0 as impossible)
#[rustfmt::skip]
const TARGET_DISTRIBUTION: [f32; 33] = [
    0.000000, 0.000000, 0.000000, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594,
    1.750000, 1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094,
    2.000000, 1.996094, 1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594,
    1.750000, 1.683594, 1.609375, 1.527344, 1.437500, 1.339844, 1.234375, 1.121094,
    1.000000
];

/// Atomic counters for adaptive piece count distribution control
struct PieceCountController {
    /// Number of positions SEEN (before filtering) with each piece count
    seen: [AtomicU64; 33],
    /// Number of positions KEPT (after filtering) with each piece count
    kept: [AtomicU64; 33],
    /// Normalized target ratios (computed once at init)
    target_ratios: [f32; 33],
    /// Counter for debug printing
    last_debug_print: AtomicU64,
}

impl PieceCountController {
    fn new() -> Self {
        // Normalize target distribution to sum to 1
        let sum: f32 = TARGET_DISTRIBUTION.iter().sum();
        let mut target_ratios = [0.0f32; 33];
        for i in 0..33 {
            target_ratios[i] = if sum > 0.0 { TARGET_DISTRIBUTION[i] / sum } else { 0.0 };
        }

        Self {
            seen: std::array::from_fn(|_| AtomicU64::new(0)),
            kept: std::array::from_fn(|_| AtomicU64::new(0)),
            target_ratios,
            last_debug_print: AtomicU64::new(0),
        }
    }

    fn print_debug_stats(&self, total_seen: u64) {
        let total_kept: u64 = self.kept.iter().map(|x| x.load(Ordering::Relaxed)).sum();

        println!(
            "\n======== Piece Count Distribution (seen: {}, kept: {}, ratio: {:.2}%) ========",
            total_seen,
            total_kept,
            100.0 * total_kept as f64 / total_seen as f64
        );
        println!(
            "{:>3} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8}",
            "PC", "Seen", "Kept", "Target%", "Actual%", "Error%"
        );
        println!("{:-<65}", "");

        for pc in 2..=32 {
            let seen = self.seen[pc].load(Ordering::Relaxed);
            let kept = self.kept[pc].load(Ordering::Relaxed);
            let target_pct = self.target_ratios[pc] * 100.0;
            let actual_pct = if total_kept > 0 { 100.0 * kept as f64 / total_kept as f64 } else { 0.0 };
            let error_pct = actual_pct - target_pct as f64;

            if seen > 0 {
                println!(
                    "{:>3} | {:>10} | {:>10} | {:>7.3}% | {:>7.3}% | {:>+7.3}%",
                    pc, seen, kept, target_pct, actual_pct, error_pct
                );
            }
        }
        println!();
    }

    fn should_keep(&self, piece_count: usize) -> bool {
        let pc = piece_count.min(32);

        // Record that we saw this piece count
        let seen_this = self.seen[pc].fetch_add(1, Ordering::Relaxed) + 1;

        let target_ratio = self.target_ratios[pc];

        // If target is 0, always skip
        if target_ratio == 0.0 {
            return false;
        }

        // Need some samples before we start controlling
        let total_seen: u64 = self.seen.iter().map(|x| x.load(Ordering::Relaxed)).sum();
        if total_seen < 10000 {
            self.kept[pc].fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Debug printing
        if DEBUG_DISTRIBUTION {
            let last_print = self.last_debug_print.load(Ordering::Relaxed);
            if total_seen >= last_print + DEBUG_PRINT_INTERVAL {
                // Try to claim this print slot (avoid multiple threads printing)
                if self
                    .last_debug_print
                    .compare_exchange(last_print, total_seen, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    self.print_debug_stats(total_seen);
                }
            }
        }

        // Find the limiting factor: min over all categories of (seen[j] / target[j])
        // This tells us the maximum "budget" we can achieve while maintaining ratios
        // Only consider piece counts that have actually been observed (skip impossible ones)
        let mut min_ratio = f64::MAX;
        for j in 0..33 {
            let seen_j = self.seen[j].load(Ordering::Relaxed) as f64;
            if self.target_ratios[j] > 0.0 && seen_j > 0.0 {
                let ratio = seen_j / (self.target_ratios[j] as f64);
                if ratio < min_ratio {
                    min_ratio = ratio;
                }
            }
        }

        // Ideal number to keep for this piece count = target[pc] * min_ratio
        // keep_prob = ideal_kept / seen = target[pc] * min_ratio / seen[pc]
        let keep_prob = (self.target_ratios[pc] as f64 * min_ratio / seen_this as f64).min(1.0);

        let keep = fast_random() < keep_prob as f32;
        if keep {
            self.kept[pc].fetch_add(1, Ordering::Relaxed);
        }
        keep
    }
}

static CONTROLLER: OnceLock<PieceCountController> = OnceLock::new();

fn get_controller() -> &'static PieceCountController {
    CONTROLLER.get_or_init(PieceCountController::new)
}

/// Simple thread-safe RNG for probabilistic filtering
static RNG_STATE: AtomicU64 = AtomicU64::new(0xDEADBEEF12345678);

fn fast_random() -> f32 {
    // xorshift64 for speed
    let mut state = RNG_STATE.load(Ordering::Relaxed);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    RNG_STATE.store(state, Ordering::Relaxed);
    (state as f32) / (u64::MAX as f32)
}

/// Custom filter that adaptively controls piece count distribution
fn piece_count_filter(board: &Board) -> bool {
    let piece_count = board.n_men() as usize;
    get_controller().should_keep(piece_count)
}

/// Eval scale used for sigmoid (same as eval_scale in training config)
const EVAL_SCALE: f32 = 160.0;

fn sigmoid(eval: f32) -> f32 {
    1.0 / (1.0 + (-eval / EVAL_SCALE).exp())
}

/// WDL-eval disagreement filter: skip positions where eval disagrees with game result
/// Positions are skipped with probability = abs(wdl - sigmoid(eval))
fn wdl_eval_disagreement_filter(eval: i16, wdl: f32) -> bool {
    let eval_wdl = sigmoid(eval as f32);
    let disagreement = (wdl - eval_wdl).abs();

    // Skip with probability equal to disagreement
    // i.e., keep with probability = 1 - disagreement
    fast_random() >= disagreement
}

fn custom_filter_pipeline(board: &Board, mv: viriformat::chess::chessmove::Move, eval: i16, wdl: f32) -> bool {
    if board.is_tactical(mv) {
        return false;
    }
    if board.in_check() {
        return false;
    }
    if !wdl_eval_disagreement_filter(eval, wdl) {
        return false;
    }
    if !piece_count_filter(board) {
        return false;
    }
    true
}

const NET_ID: &str = "bullet_r136";

fn main() {
    // network hyperparams
    let ft_size = 640;
    let l1_size = 16;
    let l2_size = 32;
    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ];

    let threat_inputs = ChessBucketsMirroredWithThreats::new(BUCKET_LAYOUT);
    let num_inputs = threat_inputs.num_inputs();

    // l0w split: PSQ features (king-bucketed + unbucketed) as i16/255, threat features as i8/255
    let factoriser_offset = threat_inputs.factorizer_base * ft_size;
    let threats_offset = threat_inputs.threat_base * ft_size;
    let save_format = [
        SavedFormat::id("l0w")
            .transform(move |_, values| {
                let king_piece_square = &values[..factoriser_offset];
                let factoriser = &values[factoriser_offset..threats_offset];
                factoriser
                    .repeat(threat_inputs.num_buckets)
                    .iter()
                    .zip(king_piece_square.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .quantise::<i16>(255)
            .round(),
        SavedFormat::id("l0w")
            .transform(move |_, values| {
                let max = 127.0 / 255.0;
                values[threats_offset..].iter().map(|&v| v.clamp(-max, max)).collect()
            })
            .quantise::<i8>(255)
            .round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255).round(),
        SavedFormat::id("l2w").transpose().round(),
        SavedFormat::id("l2b").round(),
        SavedFormat::id("l3w").transpose().round(),
        SavedFormat::id("l3b").round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(threat_inputs)
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
        .save_format(&save_format)
        .loss_fn(|output, target| {
            // asymmetric squared error: punish too high scores more
            const SKEW: f32 = 0.15;
            let e = output.sigmoid() - target;
            let relu_e = e.relu();
            e * e + SKEW * relu_e * relu_e
        })
        .build(|builder, stm, ntm, buckets| {
            // input layer weights
            let l0 = builder.new_affine("l0", num_inputs, ft_size);

            // layerstack weights
            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS * l1_size);
            let l2 = builder.new_affine("l2", l1_size * 2, NUM_OUTPUT_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);
            out = out.concat(out.abs_pow(2.0)).crelu();
            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            out
        });

    let default_ranger = RangerParams { beta1: 0.95, ..Default::default() };

    // cap l0 weights to 1.98 after factoriser is applied
    let l0_params = RangerParams { max_weight: 0.99, min_weight: -0.99, ..default_ranger };

    // allow float weights to have a large range
    let float_params = RangerParams { max_weight: 128.0, min_weight: -128.0, ..default_ranger };

    trainer.optimiser.set_params(default_ranger);
    trainer.optimiser.set_params_for_weight("l0w", l0_params);
    trainer.optimiser.set_params_for_weight("l2w", float_params);
    trainer.optimiser.set_params_for_weight("l2b", float_params);
    trainer.optimiser.set_params_for_weight("l3w", float_params);
    trainer.optimiser.set_params_for_weight("l3b", float_params);

    let stage_1_num_superbatches = 900;
    let stage_1_schedule = TrainingSchedule {
        net_id: format!("{NET_ID}-stage1"),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: stage_1_num_superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.7 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.0,
            final_superbatch: stage_1_num_superbatches,
        },
        save_rate: 100,
    };

    let stage_2_num_superbatches = 100;
    let stage_2_schedule = TrainingSchedule {
        net_id: format!("{NET_ID}-stage2"),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: stage_2_num_superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.00001,
            final_lr: 0.0,
            final_superbatch: stage_2_num_superbatches,
        },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let data_loader = ViriBinpackLoader::new(
        "..\\..\\chess\\data\\datagen3-22.viri",
        1024 * 32,
        4,
        viribinpack::ViriFilter::Custom(custom_filter_pipeline),
    );

    //trainer.load_from_checkpoint("checkpoints/bullet_r124-stage2-100");
    trainer.run(&stage_1_schedule, &settings, &data_loader);
    trainer.run(&stage_2_schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 160.0 * eval);
    }

    trainer.save_quantised(&format!("nets/{NET_ID}.nn")).unwrap();
}
