mod decode;

use anyhow::{anyhow, Context, Result};
use argh::FromArgs;
use elf::endian::AnyEndian;
use elf::ElfBytes;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::Display,
    fs::OpenOptions,
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    process,
};
use tempfile::NamedTempFile;

use crate::decode::RiscVInstr;

fn read_path(path: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(path))
}

fn deserialize_path(path: &str) -> Result<CalyxSimOutput, String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|err| format!("{err}"))?;
    serde_json::from_reader(file).map_err(|err| format!("{err}"))
}

fn read_other_memories(data: &str) -> Result<(String, usize, usize, Option<u32>), String> {
    let parts: Vec<_> = data.split(":").collect();
    if parts.len() == 3 {
        Ok((
            parts[0].to_string(),
            parts[1].parse::<usize>().map_err(|err| format!("{err}"))?,
            parts[2].parse::<usize>().map_err(|err| format!("{err}"))?,
            None,
        ))
    } else if parts.len() == 4 {
        Ok((
            parts[0].to_string(),
            parts[1].parse::<usize>().map_err(|err| format!("{err}"))?,
            parts[2].parse::<usize>().map_err(|err| format!("{err}"))?,
            Some(parts[3].parse::<u32>().map_err(|err| format!("{err}"))?),
        ))
    } else {
        Err(String::from("Wrong number of arguments "))
    }
}

/// Translate riscv assembly into a Calyx data file
#[derive(FromArgs)]
struct Cmdline {
    #[argh(subcommand)]
    nested: Commands,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Commands {
    /// encode a risc v assembly file into Calyx data format
    Encode(EncodeOpts),
    /// decode data format into human readable form
    Decode(DecodeOpts),
}

/// Encode options
#[derive(FromArgs)]
#[argh(subcommand, name = "encode")]
pub struct EncodeOpts {
    /// input file in riscv assembly format
    #[argh(positional, from_str_fn(read_path))]
    riscv_file: PathBuf,

    /// name of instruction memory
    #[argh(option, short = 'n', default = "String::from(\"insts\")")]
    name: String,

    /// riscv assembler
    #[argh(
        option,
        short = 'a',
        default = "String::from(\"riscv64-unknown-elf-as\")"
    )]
    assembler: String,

    /// other memories
    #[argh(option, from_str_fn(read_other_memories))]
    data: Vec<(String, usize, usize, Option<u32>)>,

    /// output
    #[argh(option, short = 'o', from_str_fn(read_path))]
    output: Option<PathBuf>,
}

/// Decode options
#[derive(FromArgs)]
#[argh(subcommand, name = "decode")]
pub struct DecodeOpts {
    /// input json
    #[argh(positional, from_str_fn(deserialize_path))]
    data: Option<CalyxSimOutput>,
}

#[derive(Serialize, Debug)]
struct CalyxFormat {
    numeric_type: String,
    is_signed: bool,
    width: usize,
}

#[derive(Serialize, Debug)]
struct CalyxData {
    data: Vec<u32>,
    format: CalyxFormat,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum StringOrNum {
    S(String),
    N(u32),
}

#[derive(Deserialize, Debug)]
struct CalyxSimOutput {
    cycles: u32,
    memories: HashMap<String, Vec<StringOrNum>>,
}

impl Display for StringOrNum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StringOrNum::S(str) => write!(f, "{str}"),
            StringOrNum::N(n) => write!(f, "{n}"),
        }
    }
}

fn read_le_u32(input: &mut &[u8]) -> u32 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    u32::from_le_bytes(int_bytes.try_into().unwrap())
}

fn encode(opts: EncodeOpts) -> Result<()> {
    // make a temp file for results of assembled code
    let mut tmp_file = NamedTempFile::new()?;

    // assemble input into tmp_file
    let cmd = process::Command::new(opts.assembler)
        .arg(opts.riscv_file)
        .arg("-o")
        .arg(tmp_file.path())
        .output()?;

    if !cmd.status.success() {
        return Err(anyhow!("{}", String::from_utf8(cmd.stderr)?));
    }

    // read the elf file into a Vec<u8>
    let mut buf = Vec::new();
    tmp_file.seek(SeekFrom::Start(0))?;
    tmp_file.read_to_end(&mut buf)?;

    // parse as an elf file
    let file =
        ElfBytes::<AnyEndian>::minimal_parse(buf.as_slice()).context("Failed to parse ELF")?;

    // extract .text section
    let shdr = file.section_header_by_name(".text")?.unwrap();
    let (code, _compression) = file.section_data(&shdr)?;
    let instructions: Vec<u32> = code.chunks(4).map(|mut x| read_le_u32(&mut x)).collect();

    let mut map: HashMap<String, CalyxData> = HashMap::default();
    map.insert(
        opts.name,
        CalyxData {
            data: instructions,
            format: CalyxFormat {
                numeric_type: "bitnum".to_string(),
                is_signed: false,
                width: 32,
            },
        },
    );

    for (other_mem, size, bitwidth, opt_val) in opts.data {
        map.insert(
            other_mem,
            CalyxData {
                data: vec![opt_val.unwrap_or(0); size],
                format: CalyxFormat {
                    numeric_type: "bitnum".to_string(),
                    is_signed: false,
                    width: bitwidth,
                },
            },
        );
    }

    let output_json = serde_json::to_string_pretty(&map)?;
    if let Some(output_path) = opts.output {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_path)?;
        write!(file, "{output_json}")?;
    } else {
        println!("{output_json}");
    }

    Ok(())
}

fn decode(opts: DecodeOpts) -> Result<()> {
    let data = opts
        .data
        .unwrap_or_else(|| serde_json::from_reader(io::stdin()).unwrap());

    println!("Took {} cycles", data.cycles);

    // decode instructions if we have any in the output
    println!();
    println!("== instructions ==");
    if data.memories.contains_key("insts") {
        for (idx, ele) in data.memories["insts"].iter().enumerate() {
            if let StringOrNum::N(inst) = ele {
                println!("{idx: >3}: {}", RiscVInstr::decode(*inst)?.to_string());
            }
        }
    }

    // print register file
    println!();
    println!("== registers ==");

    let reg_map: HashMap<usize, &str> = HashMap::from([
        (0, "zero"),
        (1, "ra"),
        (2, "sp"),
        (3, "gp"),
        (4, "tp"),
        (5, "t0"),
        (6, "t1"),
        (7, "t2"),
        (8, "fp"),
        (9, "s1"),
        (10, "a0"),
        (11, "a1"),
        (12, "a2"),
        (13, "a3"),
        (14, "a4"),
        (15, "a5"),
        (16, "a6"),
        (17, "a7"),
        (18, "s2"),
        (19, "s3"),
        (20, "s4"),
        (21, "s5"),
        (22, "s6"),
        (23, "s7"),
        (24, "s8"),
        (25, "s9"),
        (26, "s10"),
        (27, "s11"),
        (28, "t3"),
        (29, "t4"),
        (30, "t5"),
        (31, "t6"),
    ]);

    if data.memories.contains_key("reg_file") {
        for (id, reg) in data.memories["reg_file"].iter().enumerate() {
            println!("x{id} {: >5}: {reg}", reg_map[&id]);
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let arg: Cmdline = argh::from_env();

    match arg.nested {
        Commands::Encode(opts) => encode(opts),
        Commands::Decode(opts) => decode(opts),
    }
}
