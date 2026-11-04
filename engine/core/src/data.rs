//! Game data, loaded at start from the folder catSim's dumper writes from your own copy of the game.
//! No table from the game is compiled into catSim: every one is read here, once, from `tables.txt`
//! next to (or one folder above) the arena's mesh files.
//!
//! `tables.txt`: `[name]` starts a table; each following line is a row of tokens, `s:text` for a
//! string and eight hex digits for a 32-bit word (a float's bits or an integer). `#` starts a comment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug)]
pub struct Row {
    pub strs: Vec<&'static str>,
    pub words: Vec<u32>,
}

impl Row {
    pub fn s(&self, i: usize) -> &'static str {
        self.strs[i]
    }
    pub fn w(&self, i: usize) -> u32 {
        self.words[i]
    }
    pub fn f(&self, i: usize) -> f32 {
        f32::from_bits(self.words[i])
    }
    pub fn wn<const N: usize>(&self, at: usize) -> [u32; N] {
        std::array::from_fn(|k| self.words[at + k])
    }
    pub fn fnn<const N: usize>(&self, at: usize) -> [f32; N] {
        std::array::from_fn(|k| f32::from_bits(self.words[at + k]))
    }
}

pub struct GameData {
    pub dir: PathBuf,
    tables: HashMap<String, Vec<Row>>,
}

static DATA: OnceLock<GameData> = OnceLock::new();

fn bad(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

fn parse(text: &str, dir: PathBuf) -> std::io::Result<GameData> {
    let mut tables: HashMap<String, Vec<Row>> = HashMap::new();
    let mut current: Option<String> = None;
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with("catsim-tables") {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            tables.entry(name.to_string()).or_default();
            current = Some(name.to_string());
            continue;
        }
        let table = current.as_ref().ok_or_else(|| bad(format!("tables.txt line {}: a row before any [table]", n + 1)))?;
        let mut row = Row { strs: Vec::new(), words: Vec::new() };
        for tok in line.split_whitespace() {
            if let Some(s) = tok.strip_prefix("s:") {
                row.strs.push(Box::leak(s.to_string().into_boxed_str()));
            } else {
                row.words.push(u32::from_str_radix(tok, 16).map_err(|_| bad(format!("tables.txt line {}: bad word {tok}", n + 1)))?);
            }
        }
        tables.get_mut(table).unwrap().push(row);
    }
    Ok(GameData { dir, tables })
}

/// Load the game data from `dir` (the data folder, or its `soccar` folder). The first load wins.
pub fn load(dir: &Path) -> std::io::Result<&'static GameData> {
    if let Some(d) = DATA.get() {
        return Ok(d);
    }
    let cands = [dir.join("tables.txt"), dir.parent().map(|p| p.join("tables.txt")).unwrap_or_default()];
    let file = cands.iter().find(|p| p.is_file()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no tables.txt in {} or above it: run catSim's dumper on your copy of the game first", dir.display()),
        )
    })?;
    let data = parse(&std::fs::read_to_string(file)?, file.parent().unwrap().to_path_buf())?;
    let _ = DATA.set(data);
    Ok(DATA.get().unwrap())
}

pub fn get() -> &'static GameData {
    DATA.get().expect("catSim's game data is not loaded: point catSim at the folder its dumper wrote")
}

pub fn table(name: &str) -> &'static [Row] {
    table_opt(name).unwrap_or_else(|| panic!("catSim's game data has no [{name}] table: re-run the dumper"))
}

pub fn table_opt(name: &str) -> Option<&'static [Row]> {
    get().tables.get(name).map(|v| v.as_slice())
}
