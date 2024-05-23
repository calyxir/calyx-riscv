use anyhow::{anyhow, Result};
use bitvec::prelude::*;

pub trait BitvecConvert {
    fn to_u8(self) -> u8;
    fn to_u16(self) -> u16;
    fn to_u32(self) -> u32;
}

impl BitvecConvert for &BitSlice<u32, Lsb0> {
    fn to_u8(self) -> u8 {
        self.into_iter()
            .by_vals()
            .map(|x| if x { 1 } else { 0 })
            .enumerate()
            .fold(0u8, |acc, (idx, val)| acc + (val << idx))
    }

    fn to_u16(self) -> u16 {
        self.into_iter()
            .by_vals()
            .map(|x| if x { 1 } else { 0 })
            .enumerate()
            .fold(0u16, |acc, (idx, val)| acc + (val << idx))
    }

    fn to_u32(self) -> u32 {
        self.into_iter()
            .by_vals()
            .map(|x| if x { 1 } else { 0 })
            .enumerate()
            .fold(0u32, |acc, (idx, val)| acc + (val << idx))
    }
}

#[derive(Debug)]
pub enum RiscVInstr {
    R {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        rd: u8,
        // bits 12 - 14
        funct3: u8,
        // bits 15 - 19
        rs1: u8,
        // bits 20 - 24
        rs2: u8,
        // bits 25 - 31
        funct7: u8,
    },
    I {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        rd: u8,
        // bits 12 - 14
        funct3: u8,
        // bits 15 - 19
        rs1: u8,
        // bites 20 - 31
        imm: u16,
    },
    S {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        imm0: u8,
        // bits 12 - 14
        funct3: u8,
        // bits 15 - 19
        rs1: u8,
        // bits 20 - 24
        rs2: u8,
        // bits 25 - 31
        imm1: u8,
    },
    B {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        imm0: BitVec<u32>,
        // bits 12 - 14
        funct3: u8,
        // bits 15 - 19
        rs1: u8,
        // bits 20 - 24
        rs2: u8,
        // bits 25 - 31
        imm1: BitVec<u32>,
    },
    U {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        rd: u8,
        // 12 - 31
        imm: u32,
    },
    J {
        // bits 0 - 6
        opcode: u8,
        // bits 7 - 11
        rd: u8,
        // 12 - 31
        imm: u32,
    },
}

impl RiscVInstr {
    fn new_r(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::R {
            opcode: bits[0..7].to_u8(),
            rd: bits[7..12].to_u8(),
            funct3: bits[12..15].to_u8(),
            rs1: bits[15..20].to_u8(),
            rs2: bits[20..25].to_u8(),
            funct7: bits[25..32].to_u8(),
        }
    }

    fn new_i(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::I {
            opcode: bits[0..7].to_u8(),
            rd: bits[7..12].to_u8(),
            funct3: bits[12..15].to_u8(),
            rs1: bits[15..20].to_u8(),
            imm: bits[20..32].to_u16(),
        }
    }

    fn new_s(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::S {
            opcode: bits[0..7].to_u8(),
            imm0: bits[7..12].to_u8(),
            funct3: bits[12..15].to_u8(),
            rs1: bits[15..20].to_u8(),
            rs2: bits[20..25].to_u8(),
            imm1: bits[25..31].to_u8(),
        }
    }

    fn new_b(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::B {
            opcode: bits[0..7].to_u8(),
            imm0: bits[7..12].to_bitvec(),
            funct3: bits[12..15].to_u8(),
            rs1: bits[15..20].to_u8(),
            rs2: bits[20..25].to_u8(),
            imm1: bits[25..32].to_bitvec(),
        }
    }

    fn new_u(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::U {
            opcode: bits[0..7].to_u8(),
            rd: bits[7..12].to_u8(),
            imm: bits[12..32].to_u32(),
        }
    }

    fn new_j(bits: &BitSlice<u32>) -> Self {
        RiscVInstr::J {
            opcode: bits[0..7].to_u8(),
            rd: bits[7..12].to_u8(),
            imm: bits[12..32].to_u32(),
        }
    }

    pub fn decode(inst: u32) -> Result<RiscVInstr> {
        let bits = inst.view_bits::<Lsb0>();

        // map opcode to instruction types
        match &bits[0..7] {
            opcode if opcode == 0b0110011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_r(bits)),
            opcode if opcode == 0b0010011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_i(bits)),
            opcode if opcode == 0b0000011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_i(bits)),
            opcode if opcode == 0b0100011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_s(bits)),
            opcode if opcode == 0b1100011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_b(bits)),
            opcode if opcode == 0b1101111u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_j(bits)),
            opcode if opcode == 0b1100111u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_i(bits)),
            opcode if opcode == 0b0110111u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_u(bits)),
            opcode if opcode == 0b1110011u8.view_bits::<Lsb0>()[..7] => Ok(RiscVInstr::new_i(bits)),
            opcode => Err(anyhow!("Unknown opcode: {}", opcode)),
        }
    }
}

impl ToString for RiscVInstr {
    fn to_string(&self) -> String {
        match self {
            RiscVInstr::R {
                opcode: _,
                rd,
                funct3,
                rs1,
                rs2,
                funct7,
            } => match (funct3, funct7) {
                (0x0, 0x00) => format!("add x{rd}, x{rs1}, x{rs2}"),
                (0x0, 0x20) => format!("sub x{rd}, x{rs1}, x{rs2}"),
                (0x4, 0x00) => format!("xor x{rd}, x{rs1}, x{rs2}"),
                (0x6, 0x00) => format!("or x{rd}, x{rs1}, x{rs2}"),
                (0x7, 0x00) => format!("and x{rd}, x{rs1}, x{rs2}"),
                (0x1, 0x00) => format!("sll x{rd}, x{rs1}, x{rs2}"),
                (0x5, 0x00) => format!("srl x{rd}, x{rs1}, x{rs2}"),
                (0x5, 0x20) => format!("sra x{rd}, x{rs1}, x{rs2}"),
                (0x2, 0x00) => format!("slt x{rd}, x{rs1}, x{rs2}"),
                (0x3, 0x00) => format!("sltu x{rd}, x{rs1}, x{rs2}"),
                _ => format!("{self:?}"),
            },
            RiscVInstr::I {
                opcode,
                rd,
                funct3,
                rs1,
                imm,
            } => match (opcode, funct3) {
                (0b0010011, 0x0) => format!("addi x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x4) => format!("xori x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x6) => format!("ori x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x7) => format!("andi x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x1) => format!("slli x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x5) => format!("srli x{rd}, x{rs1}, {imm}"),
                // (0b0010011, 0x0) => format!("srai x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x2) => format!("slti x{rd}, x{rs1}, {imm}"),
                (0b0010011, 0x3) => format!("sltiu x{rd}, x{rs1}, {imm}"),
                // ========
                (0b0000011, 0x0) => format!("lb xxx"),
                (0b0000011, 0x1) => format!("lh xxx"),
                (0b0000011, 0x2) => format!("lw xxx"),
                (0b0000011, 0x4) => format!("lbu xxx"),
                (0b0000011, 0x5) => format!("lhu xxx"),
                _ => format!("{self:?}"),
            },
            RiscVInstr::S {
                opcode: _,
                imm0: _,
                funct3,
                rs1: _,
                rs2: _,
                imm1: _,
            } => match funct3 {
                _ => format!("{self:?}"),
            },
            RiscVInstr::B {
                opcode: _,
                imm0,
                funct3,
                rs1,
                rs2,
                imm1,
            } => {
                let mut reconstr_imm: BitVec<u32> = bitvec![u32, Lsb0; 0];
                // always have a zero in the lowest place
                reconstr_imm.extend_from_bitslice(&imm0[1..5]);
                reconstr_imm.extend_from_bitslice(&imm1[0..6]);
                reconstr_imm.push(imm0[0]);
                reconstr_imm.push(imm1[6]);
                let imm = reconstr_imm.as_bitslice().to_u16();
                // let reconstructed_imm = imm0[1..5];
                match funct3 {
                    0x0 => format!("beq x{rs1}, x{rs2}, {imm}"),
                    0x1 => format!("bne x{rs1}, x{rs2}, {imm}"),
                    0x4 => format!("blt x{rs1}, x{rs2}, {imm}"),
                    0x5 => format!("bge x{rs1}, x{rs2}, {imm:x}"),
                    0x6 => format!("bltu x{rs1}, x{rs2}, {imm}"),
                    0x7 => format!("bgeu x{rs1}, x{rs2}, {imm}"),
                    _ => format!("{self:?}"),
                }
            }
            RiscVInstr::U {
                opcode,
                rd: _,
                imm: _,
            } => match opcode {
                _ => format!("{self:?}"),
            },
            RiscVInstr::J {
                opcode: _,
                rd: _,
                imm: _,
            } => format!("{self:?}"),
        }
    }
}
