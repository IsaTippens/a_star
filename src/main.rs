// main.rs
// Student Name: Isa Tippens
// Student Number: 4034973

use std::{cmp::Ordering, collections::BinaryHeap, env, fs, time::Instant};

type Pos = (usize, usize);
type MagicSquare = Vec<Vec<i64>>;
type Heuristic = fn(&MagicSquare) -> i64;
type Goal = i64;
type Algorithm = fn(&mut MagicSquare, Heuristic, Goal) -> Result<State, String>;

const MAX_DEPTH: i64 = 50_000;
const DEBUG: bool = false;
// State struct

#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
enum Actions {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Hash)]
struct State {
    square: MagicSquare,
    g: i64,
    h: i64,
    pos: Pos,
    action: Option<Actions>,
    parent: Option<Box<State>>,
}

impl State {
    pub fn f(&self) -> i64 {
        self.g + self.h
    }
}

impl Eq for State {}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.square == other.square
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max heap, so we need to reverse the ordering
        self.f().cmp(&other.f()).reverse()
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.f().cmp(&other.f()).reverse())
    }
}

// END State struct

// MagicSquareSolver struct
struct MagicSquareSolver {
    h: Heuristic,
    algorithm: Algorithm,
}

impl MagicSquareSolver {
    pub fn new(algorithm: Algorithm, heuristic: Heuristic) -> MagicSquareSolver {
        return MagicSquareSolver {
            h: heuristic,
            algorithm: algorithm,
        };
    }

    pub fn solve(&mut self, square: &mut MagicSquare) {
        let goal = k(square.len() as i64);
        let start = Instant::now();
        match (self.algorithm)(square, self.h, goal) {
            Ok(s) => {
                println!("Solution found!");

                print_magic_square(&s.square);
                println!("Costs: g: {}, h: {}, f: {}", s.g, s.h, s.f());
                print_path(&s);
            }
            Err(e) => println!("{}", e),
        }
        let end = Instant::now();
        println!("Time elapsed: {:?}", end - start);
    }
}

// END MagicSquareSolver struct

fn find_pos(square: &MagicSquare, value: i64) -> Option<Pos> {
    for i in 0..square.len() {
        for j in 0..square.len() {
            if square[i][j] == value {
                return Some((j, i));
            }
        }
    }
    None
}

fn k(n: i64) -> i64 {
    // ( sum i = (1 -> n^2) => i ) / n
    let n2 = n * n;
    let sum = n2 * (n2 + 1) / 2; // arithmetic series identity
    return sum / n;
}

// Heuristics

fn h1(magic_square: &MagicSquare) -> i64 {
    let n = magic_square.len();
    let mut count = 0;
    let k_value = k(n as i64);

    for i in 0..n {
        let mut row_sum = 0;
        for j in 0..n {
            row_sum += magic_square[i][j];
        }
        if row_sum != k_value {
            count += 1;
        }
    }

    for j in 0..n {
        let mut column_sum = 0;
        for i in 0..n {
            column_sum += magic_square[i][j];
        }
        if column_sum != k_value {
            count += 1;
        }
    }

    let mut diag_sum = 0;
    for i in 0..n {
        diag_sum += magic_square[i][i];
    }
    if diag_sum != k_value {
        count += 1;
    }

    diag_sum = 0;
    for i in 0..n {
        let p = n - (i + 1);
        diag_sum += magic_square[i][p];
    }
    if diag_sum != k_value {
        count += 1
    }

    return count;
}

fn h2(magic_square: &MagicSquare) -> i64 {
    let n = magic_square.len();
    let k_value = k(n as i64);

    let mut rows_k = 0;
    for i in 0..n {
        let mut row_sum = 0;
        for j in 0..n {
            row_sum += magic_square[i][j];
        }
        rows_k += (k_value - row_sum).abs();
    }

    let mut columns_k = 0;
    for j in 0..n {
        let mut column_sum = 0;
        for i in 0..n {
            column_sum += magic_square[i][j];
        }
        columns_k += (k_value - column_sum).abs();
    }

    let mut diag_sum = 0;
    for i in 0..n {
        diag_sum += magic_square[i][i];
    }
    let diag_k = (k_value - diag_sum).abs();
    diag_sum = 0;
    for i in 0..n {
        let p = n - (i + 1);
        diag_sum += magic_square[i][p];
    }
    let diag_rev_k = (k_value - diag_sum).abs();

    return rows_k + columns_k + diag_k + diag_rev_k;
}

// END Heuristics

fn a_star(square: &mut MagicSquare, heuristic: Heuristic, goal: i64) -> Result<State, String> {
    let mut depth = 0;
    println!("Start A* algorithm");
    let mut generated_states = 0;

    let p = square.len() * square.len();
    let pos = find_pos(square, p as i64).unwrap();

    let mut open_set: BinaryHeap<State> = BinaryHeap::new();
    let mut close_set = Vec::new(); // State, (g cost, h cost, n^2 pos)
    open_set.push(State {
        square: square.to_vec(),
        g: 0,
        h: heuristic(square),
        pos: pos,
        action: None,
        parent: None,
    });

    //println!("Starting search...");

    let mut preview_count = 5;

    while !open_set.is_empty() && depth < MAX_DEPTH {
        //println!("Depth: {}", depth);

        let state = open_set.pop().unwrap(); // The while loop ensures that this is not None
        close_set.push(state.clone());

        let current_pos = state.pos;
        let current_square = &state.square;
        if preview_count > 0 || DEBUG {
            println!("Current state: {:?}", state.square);
            println!("g: {}, h: {}, f: {}", state.g, state.h, state.f());
            preview_count -= 1;
        }

        if evaluate(goal, current_square) {
            println!("Iterations: {}", depth);
            println!("Generated states: {}", generated_states);
            return Ok(state);
        }

        // Move Left
        if current_pos.0 != 0 {
            let mut temp_square: MagicSquare = (*current_square).to_vec();
            let temp_pos = move_left(&mut temp_square, current_pos);

            let new_state = State {
                square: temp_square.clone(),
                g: state.g + 1,
                h: heuristic(&temp_square),
                pos: temp_pos,
                action: Some(Actions::Left),
                parent: Some(Box::new(state.clone())),
            };

            if !close_set.contains(&new_state) {
                open_set.push(new_state);
                generated_states += 1;
            }
        }

        // Move right
        if current_pos.0 != square.len() - 1 {
            let mut temp_square: MagicSquare = (*current_square).to_vec();
            let temp_pos = move_right(&mut temp_square, current_pos);
            let new_state = State {
                square: temp_square.clone(),
                g: state.g + 1,
                h: heuristic(&temp_square),
                pos: temp_pos,
                action: Some(Actions::Right),
                parent: Some(Box::new(state.clone())),
            };
            if !close_set.contains(&new_state) {
                open_set.push(new_state);
                generated_states += 1;
            }
        }

        // Move up
        if current_pos.1 != 0 {
            let mut temp_square: MagicSquare = (*current_square).to_vec();
            let temp_pos = move_up(&mut temp_square, current_pos);
            let new_state = State {
                square: temp_square.clone(),
                g: state.g + 1,
                h: heuristic(&temp_square),
                pos: temp_pos,
                action: Some(Actions::Up),
                parent: Some(Box::new(state.clone())),
            };
            if !close_set.contains(&new_state) {
                open_set.push(new_state);
                generated_states += 1;
            }
        }

        // Move down
        if current_pos.1 != square.len() - 1 {
            let mut temp_square: MagicSquare = (*current_square).to_vec();
            let temp_pos = move_down(&mut temp_square, current_pos);
            let new_state = State {
                square: temp_square.clone(),
                g: state.g + 1,
                h: heuristic(&temp_square),
                pos: temp_pos,
                action: Some(Actions::Down),
                parent: Some(Box::new(state.clone())),
            };
            if !close_set.contains(&new_state) {
                open_set.push(new_state);
                generated_states += 1;
            }
        }
        depth += 1;
    }
    if depth >= MAX_DEPTH {
        return Err("Max interations reached!".to_string());
    }
    Err("No solution found!".to_string())
}

fn print_path(state: &State) {
    let mut current_state = state;
    let mut path = Vec::new();
    while let Some(parent) = &current_state.parent {
        let action = match current_state.action {
            Some(a) => a,
            None => break,
        };
        path.push(action);
        current_state = parent;
    }

    path.reverse();
    print!("Path: [");
    for action in path {
        print!("{:?}, ", action);
    }
    println!("]");
}

fn evaluate(goal: i64, square: &MagicSquare) -> bool {
    // Check rows
    for row in square {
        let mut sum = 0;
        for cell in row {
            sum += cell;
        }
        if sum != goal {
            return false;
        }
    }

    // Check columns
    for i in 0..square.len() {
        let mut sum = 0;
        for j in 0..square.len() {
            sum += square[j][i];
        }
        if sum != goal {
            return false;
        }
    }

    // Forward diagonal
    let mut sum = 0;
    for i in 0..square.len() {
        sum += square[i][i];
    }
    if sum != goal {
        return false;
    }

    // Reverse diagonal
    sum = 0;
    for i in 0..square.len() {
        let p = square.len() - (i + 1);
        sum += square[i][p];
    }
    if sum != goal {
        return false;
    }
    true
}



// Magic Square Methods

fn move_up(square: &mut MagicSquare, pos: Pos) -> Pos {
    let (x, y) = pos;
    let n = square.len();

    let new_y = if y == 0 { n - 1 } else { y - 1 };
    let temp = square[new_y][x];
    square[new_y][x] = square[y][x];
    square[y][x] = temp;
    (x, new_y)
}

fn move_down(square: &mut MagicSquare, pos: Pos) -> Pos {
    let (x, y) = pos;
    let n = square.len();

    let new_y = if y == n - 1 { 0 } else { y + 1 };
    let temp = square[new_y][x];
    square[new_y][x] = square[y][x];
    square[y][x] = temp;
    (x, new_y)
}

fn move_left(square: &mut MagicSquare, pos: Pos) -> Pos {
    let (x, y) = pos;
    let n = square.len();

    let new_x = if x == 0 { n - 1 } else { x - 1 };
    let temp = square[y][new_x];
    square[y][new_x] = square[y][x];
    square[y][x] = temp;
    (new_x, y)
}

fn move_right(square: &mut MagicSquare, pos: Pos) -> Pos {
    let (x, y) = pos;
    let n = square.len();

    let new_x = if x == n - 1 { 0 } else { x + 1 };
    let temp = square[y][new_x];
    square[y][new_x] = square[y][x];
    square[y][x] = temp;
    (new_x, y)
}

// END Magic Square Methods

// Utilities

fn read_file(path: &str) -> Vec<String> {
    let data = fs::read_to_string(path).expect("Failed to read file.");

    return data.lines().map(String::from).collect();
}

fn generate_magic_square(n: i64, numbers: Vec<i64>) -> MagicSquare {
    let mut square: MagicSquare = Vec::new();
    let length = n as usize;
    for i in 0..n {
        let start = length * i as usize;
        let end = length * (i + 1) as usize;
        let row = numbers[start..end].to_vec();
        square.push(row);
    }
    square
}

fn print_magic_square(square: &MagicSquare) {
    square.iter().for_each(|it| {
        print!("[");
        it.iter().for_each(|it| print!("{} ", it));
        println!("]");
    });
}

// END Utilities

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Please specifiy a file!\nmain.exe file_path.txt");
        return;
    }
    let path = &args[1];

    let lines = read_file(path);

    let n = lines[0].parse::<i64>().unwrap();
    let numbers: Vec<i64> = lines[1]
        .split_whitespace()
        .map(|s| s.parse::<i64>())
        .filter_map(Result::ok)
        .collect();

    if numbers.len() != (n * n) as usize {
        println!("Magic Square must contain only {} numbers!", { n * n });
        return;
    }

    let mut square = generate_magic_square(n, numbers);

    println!("Initial Magic Square:");
    print_magic_square(&square);

    let mut solver = MagicSquareSolver::new(a_star, h1);
    println!("Running Heuristic #1");
    solver.solve(&mut square);

    println!("Running Heuristic #2");
    solver.h = h2;
    solver.solve(&mut square);
}
