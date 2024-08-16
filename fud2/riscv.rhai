import "rtl_sim" as sim;
import "icarus" as i;
import "testbench" as tb;
import "calyx" as c;

const riscv_asm = state("riscv-asm", ["s"]);
const riscv_decode = state("riscv-decode", ["out"]);

// having to do this is not ideal, I would like to be able to reuse the one
// defined in sim
fn riscv_sim_setup(e) {
    // Data conversion to and from JSON.
    e.config_var_or("python", "python", "python3");
    e.rsrc("json-dat.py");
    e.rule("hex-data", "$python json-dat.py --from-json $in $out");
    e.rule("json-data", "$python json-dat.py --to-json $out $in");

    // Rule for simulation execution.
    e.rule(
        "sim-run",
        "./$bin +DATA=$sim_data +CYCLE_LIMIT=$cycle-limit > $out",
    );

    // More shared configuration.
    e.config_var_or("cycle-limit", "sim.cycle_limit", "500000000");
}

fn riscv_setup(e) {
   e.config_var_or("riscv-cmd", "riscv.exe", "riscv-to-calyx");
   // e.var_("riscv-cmd", "cargo run --manifest-path=../riscv-to-calyx/Cargo.toml");
   e.config_var_or("riscv-as", "riscv.assembler", "riscv64-unknown-elf-as");

   e.rule(
      "riscv-encode",
      "$riscv-cmd encode $in -a $riscv-as --data ram:16:32:1 --data reg_file:32:32 --data dbg:1:32 > $out"
   );
   e.rule("riscv-decode", "$riscv-cmd decode $in > $out");
}

fn riscv_build(e, input, output) {
   let data = `encoding.json`;
   // translate input into calyx json format
   e.build_cmd(
        [data],
        "riscv-encode",
        [input],
        []
   );
   // compile the processor into verilog
   e.build_cmd(
        ["main.sv"],
        "calyx",
        ["../src/main.futil"],
        []
   );
   e.arg("backend", "verilog");
   e.arg("args", "--disable-verify");

   // use icarus to build simulation of verilog
   e.build_cmd(
        ["riscv.exe"],
        "icarus-compile-standalone-tb",
        ["main.sv"],
        ["tb.sv", "sim_data"]
   );

   // encode .json into dat format
   e.build_cmd(
        ["sim_data"],
        "hex-data",
        [data],
        ["json-dat.py"],
   );

   // actually run the simulation
   e.build_cmd(
        ["sim.log"],
        "sim-run",
        [],
        [data, "riscv.exe"]
   );
   e.arg("sim_data", "sim_data");
   e.arg("bin", "riscv.exe");

   // convert dat back into calyx json 
   e.build_cmd(
        [output],
        "json-data",
        ["sim_data", "sim.log"],
        ["json-dat.py"]
   );
}

op(
   "riscv-simulate",
   [riscv_sim_setup, riscv_setup, i::icarus_setup, tb::standalone_setup, c::calyx_setup],
   riscv_asm,
   sim::dat, 
   riscv_build
);

rule([riscv_setup], sim::dat, riscv_decode, "riscv-decode");
