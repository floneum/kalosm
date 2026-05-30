#![allow(unused)]

use kalosm::language::{kalosm_sample, Parse, Schema};

#[derive(Parse, Schema, Clone)]
struct UnitStruct;

#[derive(Parse, Schema, Clone)]
#[parse(rename = "unit struct")]
struct RenamedUnit;
