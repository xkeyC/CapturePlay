//! Our own `NVSDK_NGX_Parameter`, for snippets that export no parameter
//! allocator (and without an NGX core to ask).
//!
//! The snippet is an MSVC build calling the interface of
//! `nvsdk_ngx_params.h` through its vtable. MSVC lays out overloaded virtual
//! functions grouped at the first declaration and in reverse declaration
//! order, so the header's Set(u64, f32, f64, u32, i32, ID3D11Resource*,
//! ID3D12Resource*, void*), Get(same, as pointers) and Reset() become the
//! slots of [`VTable`]. Values keep their type and convert on read the way
//! NGX's own block does (numbers to numbers, pointers as integers).

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Mutex;

/// `NVSDK_NGX_Result_Success` / `NVSDK_NGX_Result_Fail`.
const SUCCESS: u32 = 1;
const FAIL: u32 = 0xBAD0_0000;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Ptr(usize),
    I32(i32),
    U32(u32),
    F64(f64),
    F32(f32),
    U64(u64),
}

impl Value {
    fn as_f64(self) -> f64 {
        match self {
            Self::Ptr(v) => v as f64,
            Self::I32(v) => f64::from(v),
            Self::U32(v) => f64::from(v),
            Self::F64(v) => v,
            Self::F32(v) => f64::from(v),
            Self::U64(v) => v as f64,
        }
    }

    fn as_u64(self) -> u64 {
        match self {
            Self::Ptr(v) => v as u64,
            Self::I32(v) => v as u64,
            Self::U32(v) => u64::from(v),
            Self::U64(v) => v,
            Self::F64(_) | Self::F32(_) => self.as_f64() as u64,
        }
    }
}

type Set<T> = unsafe extern "system" fn(*mut Object, *const c_char, T);
type Get<T> = unsafe extern "system" fn(*const Object, *const c_char, *mut T) -> u32;

#[repr(C)]
struct VTable {
    set_ptr: Set<*mut c_void>,
    set_d3d12: Set<*mut c_void>,
    set_d3d11: Set<*mut c_void>,
    set_i32: Set<i32>,
    set_u32: Set<u32>,
    set_f64: Set<f64>,
    set_f32: Set<f32>,
    set_u64: Set<u64>,
    get_ptr: Get<*mut c_void>,
    get_d3d12: Get<*mut c_void>,
    get_d3d11: Get<*mut c_void>,
    get_i32: Get<i32>,
    get_u32: Get<u32>,
    get_f64: Get<f64>,
    get_f32: Get<f32>,
    get_u64: Get<u64>,
    reset: unsafe extern "system" fn(*mut Object),
}

/// The C++ object: vtable pointer first, then our state.
#[repr(C)]
struct Object {
    vtable: &'static VTable,
    values: Mutex<HashMap<CString, Value>>,
}

/// # Safety
/// `this` is an [`Object`] and `name` a NUL-terminated string (NGX contract).
unsafe fn store(this: *const Object, name: *const c_char, v: Value) {
    if name.is_null() {
        return;
    }
    // SAFETY: per the function contract.
    let (obj, key) = unsafe { (&*this, CStr::from_ptr(name).to_owned()) };
    obj.values
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key, v);
}

/// # Safety
/// As [`store`].
unsafe fn load(this: *const Object, name: *const c_char) -> Option<Value> {
    if name.is_null() {
        return None;
    }
    // SAFETY: per the function contract.
    let (obj, key) = unsafe { (&*this, CStr::from_ptr(name)) };
    obj.values
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(key)
        .copied()
}

macro_rules! setter {
    ($name:ident, $t:ty, $wrap:expr) => {
        unsafe extern "system" fn $name(this: *mut Object, name: *const c_char, v: $t) {
            // SAFETY: called by NGX on our object.
            unsafe { store(this, name, $wrap(v)) }
        }
    };
}

macro_rules! getter {
    ($name:ident, $t:ty, $conv:expr) => {
        unsafe extern "system" fn $name(
            this: *const Object,
            name: *const c_char,
            out: *mut $t,
        ) -> u32 {
            // SAFETY: called by NGX on our object with a valid out-pointer.
            match unsafe { load(this, name) } {
                Some(v) if !out.is_null() => {
                    // SAFETY: as above.
                    unsafe { out.write($conv(v)) };
                    SUCCESS
                }
                _ => FAIL,
            }
        }
    };
}

setter!(set_ptr, *mut c_void, |v: *mut c_void| Value::Ptr(
    v as usize
));
setter!(set_i32, i32, Value::I32);
setter!(set_u32, u32, Value::U32);
setter!(set_f64, f64, Value::F64);
setter!(set_f32, f32, Value::F32);
setter!(set_u64, u64, Value::U64);
getter!(get_ptr, *mut c_void, |v: Value| v.as_u64() as usize
    as *mut c_void);
getter!(get_i32, i32, |v: Value| match v {
    Value::F32(_) | Value::F64(_) => v.as_f64() as i32,
    _ => v.as_u64() as i32,
});
getter!(get_u32, u32, |v: Value| match v {
    Value::F32(_) | Value::F64(_) => v.as_f64() as u32,
    _ => v.as_u64() as u32,
});
getter!(get_f64, f64, Value::as_f64);
getter!(get_f32, f32, |v: Value| v.as_f64() as f32);
getter!(get_u64, u64, Value::as_u64);

unsafe extern "system" fn reset(this: *mut Object) {
    // SAFETY: called by NGX on our object.
    unsafe { &*this }
        .values
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

static VTABLE: VTable = VTable {
    set_ptr,
    set_d3d12: set_ptr,
    set_d3d11: set_ptr,
    set_i32,
    set_u32,
    set_f64,
    set_f32,
    set_u64,
    get_ptr,
    get_d3d12: get_ptr,
    get_d3d11: get_ptr,
    get_i32,
    get_u32,
    get_f64,
    get_f32,
    get_u64,
    reset,
};

/// A parameter block the snippet can read and write.
pub struct Params {
    object: Box<Object>,
}

impl Params {
    pub fn new() -> Self {
        Self {
            object: Box::new(Object {
                vtable: &VTABLE,
                values: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The `NVSDK_NGX_Parameter*` to hand to NGX.
    pub fn raw(&self) -> *mut c_void {
        (&raw const *self.object).cast_mut().cast()
    }

    fn put(&self, name: &CStr, v: Value) {
        // SAFETY: our own object and a NUL-terminated name.
        unsafe { store(&raw const *self.object, name.as_ptr(), v) }
    }

    pub fn set_u32(&self, name: &CStr, v: u32) {
        self.put(name, Value::U32(v));
    }

    pub fn set_i32(&self, name: &CStr, v: i32) {
        self.put(name, Value::I32(v));
    }

    pub fn set_f32(&self, name: &CStr, v: f32) {
        self.put(name, Value::F32(v));
    }

    pub fn set_ptr(&self, name: &CStr, v: *mut c_void) {
        self.put(name, Value::Ptr(v as usize));
    }

    /// What the snippet left under `name` (diagnostics).
    pub fn get_u32(&self, name: &CStr) -> Option<u32> {
        let mut v = 0;
        // SAFETY: our own object, through the vtable the snippet uses.
        let r =
            unsafe { (self.object.vtable.get_u32)(&raw const *self.object, name.as_ptr(), &mut v) };
        (r == SUCCESS).then_some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_like_ngx() {
        let p = Params::new();
        p.set_i32(c"a", 3);
        p.set_f32(c"b", 2.5);
        assert_eq!(p.get_u32(c"a"), Some(3));
        assert_eq!(p.get_u32(c"b"), Some(2));
        assert_eq!(p.get_u32(c"missing"), None);
    }
}
