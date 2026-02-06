use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{Ranger, RangerParams},
    },
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

/// Lasso regression coefficient for L1 layer activations
/// Penalizes large activations with: coefficient * avg(|activations|)
const L1_LASSO_COEFFICIENT: f32 = 0.005;

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

const NET_ID: &str = "bullet_r116-768x8hm-1536-dp-pw-16-da-32-1x8";

fn main() {
    // network hyperparams
    const FT_SIZE: usize = 1536;
    const L1_SIZE: usize = 16;
    const L2_SIZE: usize = 32;
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
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let save_format = [
        SavedFormat::id("l0w")
            .transform(|store, weights| {
                let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .quantise::<i16>(255)
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
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(FT_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, FT_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1 = builder.new_affine("l1", FT_SIZE, NUM_OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE * 2, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, NUM_OUTPUT_BUCKETS);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // lasso regularization to encourage sparsity
            // sum across features using matmul with ones, then average across batch
            let ones = builder.new_constant(Shape::new(1, FT_SIZE), &[1.0; FT_SIZE]);
            let l0_lasso = ones.matmul(out) * (L1_LASSO_COEFFICIENT / FT_SIZE as f32);

            // layerstack inference
            out = l1.forward(out).select(buckets);
            out = out.concat(out.abs_pow(2.0)).crelu();
            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            // squared error loss + regularization
            let loss = out.sigmoid().squared_error(targets) + l0_lasso;
            (out, loss)
        });

    // cap l1 weights to 1.98 after factoriser is applied
    let l0_params = RangerParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };

    // allow float weights to have a large range
    let float_params = RangerParams { max_weight: 128.0, min_weight: -128.0, ..Default::default() };

    trainer.optimiser.set_params_for_weight("l0w", l0_params);
    trainer.optimiser.set_params_for_weight("l0f", l0_params);
    trainer.optimiser.set_params_for_weight("l2w", float_params);
    trainer.optimiser.set_params_for_weight("l2b", float_params);
    trainer.optimiser.set_params_for_weight("l3w", float_params);
    trainer.optimiser.set_params_for_weight("l3b", float_params);

    let num_superbatches = 1000;
    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: num_superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.7 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: num_superbatches },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let data_loader = ViriBinpackLoader::new(
        "..\\..\\chess\\data\\datagen3-22.viri",
        1024 * 32,
        4,
        viribinpack::ViriFilter::Custom(custom_filter_pipeline),
    );

    //trainer.load_from_checkpoint(...);
    trainer.run(&schedule, &settings, &data_loader);

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

    trainer.save_quantised(&format!("nets/{NET_ID}-e{num_superbatches}.nn")).unwrap();
}
