use pyo3::prelude::*;

#[cfg(target_os = "windows")]
mod sys {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn query_memory() -> (u64, u64, u64, f64) {
        unsafe {
            let mut status = MemoryStatusEx {
                dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
                dw_memory_load: 0,
                ull_total_phys: 0,
                ull_avail_phys: 0,
                ull_total_page_file: 0,
                ull_avail_page_file: 0,
                ull_total_virtual: 0,
                ull_avail_virtual: 0,
                ull_avail_extended_virtual: 0,
            };
            if GlobalMemoryStatusEx(&mut status) != 0 {
                let total = status.ull_total_phys;
                let avail = status.ull_avail_phys;
                let used = total.saturating_sub(avail);
                let pct = if total > 0 {
                    (used as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                (total, avail, used, pct)
            } else {
                (0, 0, 0, 0.0)
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    pub fn query_memory() -> (u64, u64, u64, f64) {
        let mut total_kb: u64 = 0;
        let mut avail_kb: u64 = 0;
        if let Ok(file) = File::open("/proc/meminfo") {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if line.starts_with("MemTotal:") {
                    total_kb = parse_kb(&line);
                } else if line.starts_with("MemAvailable:") {
                    avail_kb = parse_kb(&line);
                }
            }
        }
        let total = total_kb * 1024;
        let avail = avail_kb * 1024;
        let used = total.saturating_sub(avail);
        let pct = if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        (total, avail, used, pct)
    }

    fn parse_kb(line: &str) -> u64 {
        line.split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod sys {
    pub fn query_memory() -> (u64, u64, u64, f64) {
        (0, 0, 0, 0.0)
    }
}

/// Consulta memoria física del sistema: (total_bytes, avail_bytes, used_bytes, percent)
#[pyfunction]
pub fn get_system_memory() -> (u64, u64, u64, f64) {
    sys::query_memory()
}
