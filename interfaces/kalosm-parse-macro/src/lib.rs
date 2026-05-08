use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote_spanned;
use quote::{format_ident, quote, ToTokens};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use syn::meta::ParseNestedMeta;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{ext::IdentExt, parse_macro_input, DeriveInput, Field, Ident, LitStr};
use syn::{DataEnum, Fields, FieldsNamed, LitInt, Path, TypePath, Variant};

/// Derive a default JSON parser for a unit value, struct or enum.
///
/// # Examples
///
/// You can derive a parser for a struct with fields that implement the `Parse` trait:
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Debug, Clone, PartialEq)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let parser = Person::new_parser();
/// let state = parser.create_parser_state();
/// let person = parser
///     .parse(&state, b"{ \"name\": \"John\", \"age\": 30 } ")
///     .unwrap()
///     .unwrap_finished();
/// assert_eq!(person.name, "John");
/// assert_eq!(person.age, 30);
/// ```
///
/// Or an enum with unit variants:
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Debug, Clone, PartialEq)]
/// enum Color {
///     Red,
///     Blue,
///     Green,
/// }
///
/// let parser = Color::new_parser();
/// let state = parser.create_parser_state();
/// let color = parser.parse(&state, b"\"Red\" ").unwrap().unwrap_finished();
/// assert_eq!(color, Color::Red);
/// ```
///
/// You can even derive Parse for an enum with data variants:
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Debug, Clone, PartialEq)]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
///
/// let parser = Action::new_parser();
/// let state = parser.create_parser_state();
/// let action = parser
///     .parse(
///         &state,
///         b"{ \"type\": \"Search\", \"data\": { \"query\": \"my query\" } } ",
///     )
///     .unwrap()
///     .unwrap_finished();
/// assert_eq!(
///     action,
///     Action::Search {
///         query: "my query".to_string()
///     }
/// );
/// ```
///
/// ## Attributes
///
/// The `#[parse]` attribute modifies the default behavior of the parser. It can be used in the following forms:
///
/// - `#[parse(rename = "name")]` renames the field or type to `name` (defaults to the field name)
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Clone)]
/// struct Person {
///     #[parse(rename = "full name")]
///     name: String,
///     age: u32,
/// }
///
/// #[derive(Parse, Schema, Clone)]
/// enum Color {
///     #[parse(rename = "red")]
///     Red,
///     Blue,
///     Green,
/// }
/// ```
///
/// - `#[parse(with = expression)]` uses the expression to parse the field (defaults to the parser provided by the `Parse` implementation for the field type)
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Clone)]
/// struct Person {
///     #[parse(with = StringParser::new(1..=10))]
///     name: String,
///     age: u32,
/// }
/// ```
///
/// - `#[parse(tag = "tag")]` changes the name of the tag for enum variants (defaults to "type")
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Clone)]
/// #[parse(tag = "action")]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
/// ```
///
/// - `#[parse(content = "content")]` changes the name of the content for enum variants (defaults to "data")
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Schema, Clone)]
/// #[parse(content = "arguments")]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
/// ```
#[proc_macro_derive(Parse, attributes(parse))]
pub fn derive_parse(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(fields) => {
                let ty = input.ident;
                if fields.named.is_empty() {
                    return TokenStream::from(impl_unit_parser(
                        &input.attrs,
                        &ty,
                        quote! { Self {} },
                    ));
                }
                let struct_parser = match StructParser::new(input.attrs, fields, ty) {
                    Ok(parser) => parser,
                    Err(err) => return err.to_compile_error().into(),
                };

                TokenStream::from(struct_parser.parser())
            }
            syn::Fields::Unit => {
                let ty = input.ident;
                TokenStream::from(impl_unit_parser(&input.attrs, &ty, quote! { Self }))
            }
            _ => syn::Error::new(
                input.ident.span(),
                "Only structs with named fields are supported",
            )
            .to_compile_error()
            .into(),
        },
        syn::Data::Enum(data) => {
            let ty = input.ident;
            if data.variants.is_empty() {
                return syn::Error::new(ty.span(), "Enums with no variants are not supported")
                    .to_compile_error()
                    .into();
            }

            let has_fields = data
                .variants
                .iter()
                .any(|variant| !matches!(&variant.fields, syn::Fields::Unit));

            if has_fields {
                match EnumParser::new(input.attrs, data, ty)
                    .and_then(|parser| parser.quote_parser())
                {
                    Ok(parser) => parser,
                    Err(err) => err.to_compile_error(),
                }
            } else {
                unit_enum_parser(input.attrs, data, ty)
            }
            .into()
        }
        _ => syn::Error::new(
            input.ident.span(),
            "Only structs and unit value enums are supported",
        )
        .to_compile_error()
        .into(),
    }
}

#[proc_macro_derive(Schema, attributes(parse))]
pub fn derive_schema(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(fields) => {
                let ty = input.ident;
                if fields.named.is_empty() {
                    return TokenStream::from(unit_schema(&input.attrs, &ty));
                }
                let struct_parser = match StructParser::new(input.attrs, fields, ty) {
                    Ok(parser) => parser,
                    Err(err) => return err.to_compile_error().into(),
                };

                TokenStream::from(struct_parser.quote_schema())
            }
            syn::Fields::Unit => {
                let ty = input.ident;
                TokenStream::from(unit_schema(&input.attrs, &ty))
            }
            _ => syn::Error::new(
                input.ident.span(),
                "Only structs with named fields are supported",
            )
            .to_compile_error()
            .into(),
        },
        syn::Data::Enum(data) => {
            let ty = input.ident;
            if data.variants.is_empty() {
                return syn::Error::new(ty.span(), "Enums with no variants are not supported")
                    .to_compile_error()
                    .into();
            }

            let has_fields = data
                .variants
                .iter()
                .any(|variant| !matches!(&variant.fields, syn::Fields::Unit));

            if has_fields {
                match EnumParser::new(input.attrs, data, ty)
                    .and_then(|parser| parser.quote_schema())
                {
                    Ok(parser) => parser,
                    Err(err) => err.to_compile_error(),
                }
            } else {
                unit_enum_schema(data, ty)
            }
            .into()
        }
        _ => syn::Error::new(
            input.ident.span(),
            "Only structs and unit value enums are supported",
        )
        .to_compile_error()
        .into(),
    }
}

struct StructParser {
    attributes: Vec<syn::Attribute>,
    ty: Ident,
    name: String,
    fields: FieldsParser,
}

impl StructParser {
    fn new(attributes: Vec<syn::Attribute>, fields: FieldsNamed, ty: Ident) -> syn::Result<Self> {
        let named = fields.named.into_iter().collect::<Vec<_>>();

        let mut name = ty.unraw().to_string();
        for attr in &attributes {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if let Some(value) = parse_rename_attribute(&meta)? {
                        name = value.value();
                    } else {
                        return Err(meta.error("expected `rename`"));
                    }
                    Ok(())
                })?;
            }
        }

        Ok(Self {
            attributes,
            name,
            ty,
            fields: FieldsParser::new(&named)?,
        })
    }

    fn parser(&self) -> TokenStream2 {
        let field_names = self
            .fields
            .fields
            .iter()
            .map(|f| f.field.ident.as_ref().unwrap());
        let construct = quote! {
            Self {
                #(
                    #field_names
                ),*
            }
        };

        let parser = match self.fields.parser(construct) {
            Ok(parser) => parser,
            Err(err) => return err.to_compile_error(),
        };

        let ty = &self.ty;

        quote! {
            impl kalosm_sample::Parse for #ty {
                fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                    #parser
                }
            }
        }
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        let title = &self.name;
        let ty = &self.ty;
        let description = doc_comment(&self.attributes);
        let description = description.map(|description| quote! { .with_description(#description) });
        let schema = self.fields.quote_schema();

        quote! {
            impl kalosm_sample::Schema for #ty {
                fn schema() -> kalosm_sample::SchemaType {
                    kalosm_sample::SchemaType::Object(
                        #schema
                        .with_title(#title)
                        #description
                    )
                }
            }
        }
    }
}

fn quote_fields(fields: Fields) -> TokenStream2 {
    match fields {
        Fields::Named(fields) => {
            let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());
            quote! {
                {
                    #(
                        #field_names
                    ),*
                }
            }
        }
        Fields::Unnamed(fields) => {
            let field_names = (0..fields.unnamed.len()).map(|i| format_ident!("data{}", i));
            quote! {
                (
                    #(
                        #field_names
                    ),*
                )
            }
        }
        Fields::Unit => {
            quote! {}
        }
    }
}

fn impl_unit_parser(attrs: &[syn::Attribute], ty: &Ident, construct: TokenStream2) -> TokenStream2 {
    let unit_parser = unit_parser(attrs, ty);
    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                kalosm_sample::ParserExt::map_output(
                    #unit_parser,
                    |_| #construct
                )
            }
        }
    }
}

fn unit_schema(attrs: &[syn::Attribute], ty: &Ident) -> TokenStream2 {
    let name = match unit_parse_literal_name(attrs, ty) {
        Ok(name) => name,
        Err(err) => return err.to_compile_error(),
    };

    quote! {
        impl kalosm_sample::Schema for #ty {
            fn schema() -> kalosm_sample::SchemaType {
                kalosm_sample::SchemaType::Enum(kalosm_sample::EnumSchema::new([
                    kalosm_sample::SchemaLiteral::String(#name.to_string())
                ]))
            }
        }
    }
}

fn unit_parse_literal(attrs: &[syn::Attribute], ty: &Ident, unquoted: bool) -> syn::Result<String> {
    let ty_string = unit_parse_literal_name(attrs, ty)?;

    Ok(if unquoted {
        ty_string
    } else {
        format!("\"{ty_string}\"")
    })
}

fn unit_parse_literal_name(attrs: &[syn::Attribute], ty: &Ident) -> syn::Result<String> {
    // Look for #[parse(rename = "name")] attribute
    let mut ty_string = ty.unraw().to_string();
    for attr in attrs.iter() {
        if attr.path().is_ident("parse") {
            attr.parse_nested_meta(|meta| {
                if let Some(value) = parse_rename_attribute(&meta)? {
                    ty_string = value.value();
                    Ok(())
                } else {
                    Err(meta.error("expected `rename`"))
                }
            })?;
        }
    }

    Ok(ty_string)
}

fn unit_parser(attrs: &[syn::Attribute], ty: &Ident) -> TokenStream2 {
    let ty_string = match unit_parse_literal(attrs, ty, false) {
        Ok(ty_string) => ty_string,
        Err(err) => return err.to_compile_error(),
    };
    let ty_string = LitStr::new(&ty_string, ty.span());
    quote! {
        kalosm_sample::LiteralParser::new(#ty_string)
    }
}

struct EnumParser {
    ty: Ident,
    tag: String,
    data: String,
    variants: Vec<EnumVariant>,
}

impl EnumParser {
    fn new(attrs: Vec<syn::Attribute>, data: DataEnum, ty: Ident) -> syn::Result<Self> {
        // Look for the tag and content attributes within the #[parse] attribute
        let mut tag = "type".to_string();
        let mut content = "data".to_string();
        for attr in attrs.iter() {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("tag") {
                        let value = meta
                            .value()
                            .and_then(|value| value.parse::<syn::LitStr>())?;
                        tag = value.value();
                        Ok(())
                    } else if meta.path.is_ident("content") {
                        let value = meta
                            .value()
                            .and_then(|value| value.parse::<syn::LitStr>())?;
                        content = value.value();
                        Ok(())
                    } else {
                        Err(meta.error("expected `tag` or `content`"))
                    }
                })?;
            }
        }

        let variants = data
            .variants
            .iter()
            .map(EnumVariant::new)
            .collect::<syn::Result<_>>()?;

        Ok(EnumParser {
            ty,
            tag,
            data: content,
            variants,
        })
    }

    fn quote_parser(&self) -> syn::Result<TokenStream2> {
        let tag = &self.tag;
        let ty = &self.ty;
        let content = &self.data;
        let mut parser = None;

        for variant in &self.variants {
            let parse_variant = variant.quote_parser(content)?;
            match &mut parser {
                Some(current) => {
                    *current = quote! {
                        kalosm_sample::ParserExt::or(
                            #current,
                            #parse_variant
                        )
                    };
                }
                None => {
                    parser = Some(parse_variant);
                }
            }
        }

        let struct_start = format!("{{ \"{tag}\": \"");

        Ok(quote! {
            impl kalosm_sample::Parse for #ty {
                fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                    kalosm_sample::ParserExt::then_literal(
                        kalosm_sample::ParserExt::ignore_output_then(
                            kalosm_sample::LiteralParser::from(#struct_start),
                            #parser
                        ),
                        r#" }"#
                    )
                }
            }
        })
    }

    fn quote_schema(&self) -> syn::Result<proc_macro2::TokenStream> {
        let tag = &self.tag;
        let content = &self.data;
        let ty = &self.ty;

        let variants: Vec<_> = self
            .variants
            .iter()
            .map(|variant| {
                let variant_name = &variant.name;
                let variant_parser = variant.quote_schema(tag, content, variant_name)?;
                Ok(quote! {
                    #variant_parser
                })
            })
            .collect::<syn::Result<_>>()?;

        Ok(quote! {
            impl kalosm_sample::Schema for #ty {
                fn schema() -> kalosm_sample::SchemaType {
                    kalosm_sample::SchemaType::AnyOf(
                        kalosm_sample::AnyOfSchema::new([
                            #(#variants),*
                        ])
                    )
                }
            }
        })
    }
}

struct EnumVariant {
    variant: Variant,
    name: String,
    ty: EnumVariantType,
}

impl EnumVariant {
    fn new(variant: &Variant) -> syn::Result<Self> {
        let variant_ident = &variant.ident;
        let mut variant_name = variant_ident.unraw().to_string();
        // Look for #[parse(rename = "name")] attribute
        for attr in variant.attrs.iter() {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if let Some(value) = parse_rename_attribute(&meta)? {
                        variant_name = value.value();
                        Ok(())
                    } else {
                        Err(meta.error("expected `rename`"))
                    }
                })?;
            }
        }

        let parse_variant = match &variant.fields {
            syn::Fields::Named(fields) => {
                EnumVariantType::Struct(StructEnumVariantParser::new(fields)?)
            }
            syn::Fields::Unnamed(fields) => {
                let field_vec = fields.unnamed.iter().collect::<Vec<_>>();
                let [inner] = *field_vec else {
                    return Err(syn::Error::new(
                        variant.ident.span(),
                        "Unnamed enum variants with more or less than one field are not supported",
                    ));
                };

                EnumVariantType::Tuple(Box::new(TupleEnumVariantParser::new(inner)))
            }
            // If this is a unit variant, we can just parse the type
            syn::Fields::Unit => EnumVariantType::Unit(UnitEnumVariantParser::new()),
        };

        Ok(Self {
            variant: variant.clone(),
            name: variant_name,
            ty: parse_variant,
        })
    }

    fn construct_variant(&self) -> TokenStream2 {
        let fields = quote_fields(self.variant.fields.clone());
        let variant_ident = &self.variant.ident;
        quote! {
            Self::#variant_ident #fields
        }
    }

    fn quote_parser(&self, content_name: &str) -> syn::Result<TokenStream2> {
        let construct_variant = self.construct_variant();
        match &self.ty {
            EnumVariantType::Struct(parser) => {
                parser.quote_parser(&self.name, content_name, construct_variant)
            }
            EnumVariantType::Tuple(parser) => {
                parser.quote_parser(&self.name, content_name, construct_variant)
            }
            EnumVariantType::Unit(parser) => parser.quote_parser(&self.name, construct_variant),
        }
    }

    fn quote_schema(
        &self,
        tag: &str,
        content: &str,
        variant_name: &str,
    ) -> syn::Result<proc_macro2::TokenStream> {
        match &self.ty {
            EnumVariantType::Struct(parser) => parser.quote_schema(tag, content, variant_name),
            EnumVariantType::Tuple(parser) => parser.quote_schema(tag, content, variant_name),
            EnumVariantType::Unit(parser) => parser.quote_schema(tag, variant_name),
        }
    }
}

enum EnumVariantType {
    Unit(UnitEnumVariantParser),
    Tuple(Box<TupleEnumVariantParser>),
    Struct(StructEnumVariantParser),
}

struct UnitEnumVariantParser {}

impl UnitEnumVariantParser {
    fn new() -> Self {
        Self {}
    }

    fn quote_parser(
        &self,
        variant_name: &str,
        construct_variant: TokenStream2,
    ) -> syn::Result<TokenStream2> {
        let lit_str_name = LitStr::new(&format!("{variant_name}\""), Span::call_site());
        Ok(quote! {
            kalosm_sample::ParserExt::map_output(
                kalosm_sample::LiteralParser::from(#lit_str_name),
                |_| #construct_variant
            )
        })
    }

    fn quote_schema(&self, tag: &str, variant_name: &str) -> syn::Result<proc_macro2::TokenStream> {
        Ok(quote! {
            kalosm_sample::SchemaType::Object(
                kalosm_sample::JsonObjectSchema::new([
                    kalosm_sample::JsonPropertySchema::new(
                        #tag,
                        kalosm_sample::SchemaType::Enum(
                            kalosm_sample::EnumSchema::new([
                                kalosm_sample::SchemaLiteral::String(#variant_name.to_string())
                            ])
                        )
                    )
                    .with_required(true)
                ])
            )
        })
    }
}

struct StructEnumVariantParser {
    fields: FieldsParser,
}

impl StructEnumVariantParser {
    fn new(fields: &FieldsNamed) -> syn::Result<Self> {
        let fields = fields.named.iter().cloned().collect::<Vec<_>>();
        Ok(Self {
            fields: FieldsParser::new(&fields)?,
        })
    }

    fn quote_parser(
        &self,
        variant_name: &str,
        content_name: &str,
        construct_variant: TokenStream2,
    ) -> syn::Result<TokenStream2> {
        let parse_name_and_data = LitStr::new(
            &format!("{variant_name}\", \"{content_name}\": "),
            Span::call_site(),
        );
        let field_parser = self.fields.parser(construct_variant)?;
        Ok(quote! {
            kalosm_sample::ParserExt::ignore_output_then(
                kalosm_sample::LiteralParser::from(#parse_name_and_data),
                #field_parser
            )
        })
    }

    fn quote_schema(
        &self,
        tag: &str,
        content: &str,
        variant_name: &str,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let variant_parser = self.fields.quote_schema();
        Ok(quote! {
            kalosm_sample::SchemaType::Object(
                kalosm_sample::JsonObjectSchema::new([
                    kalosm_sample::JsonPropertySchema::new(
                        #tag,
                        kalosm_sample::SchemaType::Enum(
                            kalosm_sample::EnumSchema::new([
                                kalosm_sample::SchemaLiteral::String(#variant_name.to_string())
                            ])
                        )
                    )
                    .with_required(true),
                    kalosm_sample::JsonPropertySchema::new(
                        #content,
                        kalosm_sample::SchemaType::Object(
                            #variant_parser
                        )
                    )
                    .with_required(true)
                ])
            )
        })
    }
}

struct TupleEnumVariantParser {
    field: Field,
}

impl TupleEnumVariantParser {
    fn new(fields: &Field) -> Self {
        Self {
            field: fields.clone(),
        }
    }

    fn quote_parser(
        &self,
        variant_name: &str,
        content: &str,
        construct_variant: TokenStream2,
    ) -> syn::Result<TokenStream2> {
        let parse_name_and_data = LitStr::new(
            &format!("{variant_name}\", \"{content}\": "),
            Span::call_site(),
        );
        let ty = &self.field.ty;
        Ok(quote! {
            kalosm_sample::ParserExt::map_output(
                kalosm_sample::ParserExt::ignore_output_then(
                    kalosm_sample::LiteralParser::from(#parse_name_and_data),
                    <#ty as kalosm_sample::Parse>::new_parser()
                ),
                |data0| #construct_variant
            )
        })
    }

    fn quote_schema(
        &self,
        tag: &str,
        content: &str,
        variant_name: &str,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let ty = &self.field.ty;
        Ok(quote! {
            kalosm_sample::SchemaType::Object(
                kalosm_sample::JsonObjectSchema::new([
                    kalosm_sample::JsonPropertySchema::new(
                        #tag,
                        kalosm_sample::SchemaType::Enum(
                            kalosm_sample::EnumSchema::new([
                                kalosm_sample::SchemaLiteral::String(#variant_name.to_string())
                            ])
                        )
                    )
                    .with_required(true),
                    kalosm_sample::JsonPropertySchema::new(
                        #content,
                        <#ty as kalosm_sample::Schema>::schema()
                    )
                    .with_required(true)
                ])
            )
        })
    }
}

fn unit_enum_parser(attrs: Vec<syn::Attribute>, data: DataEnum, ty: Ident) -> TokenStream2 {
    // We can derive an efficient state machine for unit enums
    let parser_state = format_ident!("{}ParserState", ty);

    // Look for #[parse(unquoted)] on the enum
    let mut unquoted = false;
    for attr in attrs.iter() {
        if attr.path().is_ident("parse") {
            let result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("unquoted") {
                    unquoted = true;
                    return Ok(());
                }
                Err(meta.error("expected `unquoted`"))
            });
            if let Err(err) = result {
                return err.to_compile_error();
            }
        }
    }

    let mut parse_construction_map = HashMap::new();
    for variant in data.variants.iter() {
        let variant_name = &variant.ident;
        let fields = &variant.fields;
        let construct_variant = quote! {
            #ty::#variant_name #fields
        };
        let literal_string = match unit_parse_literal(&variant.attrs, variant_name, unquoted) {
            Ok(literal_string) => literal_string,
            Err(err) => return err.to_compile_error(),
        };
        parse_construction_map.insert(literal_string.as_bytes().to_vec(), construct_variant);
    }

    let mut prefix_state_map = HashMap::new();
    let mut max_state = 0usize;
    for bytes in parse_construction_map.keys() {
        for i in 0..bytes.len() + 1 {
            let prefix = &bytes[..i];
            if prefix_state_map.contains_key(prefix) {
                continue;
            }
            prefix_state_map.insert(prefix, max_state);
            max_state += 1;
        }
    }

    let mut parse_states = Vec::new();
    for (state_prefix, state) in &prefix_state_map {
        let state = LitInt::new(&state.to_string(), ty.span());

        let mut next_bytes = Vec::new();
        for (next_state_prefix, next_state) in &prefix_state_map {
            if let Some(&[byte]) = next_state_prefix.strip_prefix(*state_prefix) {
                let next_state = LitInt::new(&next_state.to_string(), ty.span());

                next_bytes.push(quote! {
                    #byte => state = #parser_state(#next_state),
                });
            }
        }

        let unrecognized_byte = if let Some(constructor) = parse_construction_map.get(*state_prefix)
        {
            quote! {
                return kalosm_sample::ParseResult::Ok(kalosm_sample::ParseStatus::Finished {
                    result: #constructor,
                    remaining: &input[i..],
                })
            }
        } else {
            quote! {
                return kalosm_sample::ParseResult::Err(kalosm_sample::ParserError::msg("Unrecognized byte"))
            }
        };

        if !next_bytes.is_empty() {
            parse_states.push(quote! {
                #state => match byte {
                    #(#next_bytes)*
                    _ => #unrecognized_byte,
                },
            });
        } else if parse_construction_map.contains_key(*state_prefix) {
            parse_states.push(quote! {
                #state => #unrecognized_byte,
            });
        }
    }

    let mut match_required_next = Vec::new();
    for (state_prefix, state) in &prefix_state_map {
        let state = LitInt::new(&state.to_string(), ty.span());
        let mut required_next = Vec::new();
        let mut current_prefix = state_prefix.to_vec();
        loop {
            let mut valid_next_bytes = Vec::new();
            for prefix in prefix_state_map.keys() {
                if let Some(&[byte]) = prefix.strip_prefix(current_prefix.as_slice()) {
                    valid_next_bytes.push(byte);
                }
            }
            if let [byte] = *valid_next_bytes {
                current_prefix.push(byte);
                required_next.push(byte);
            } else {
                break;
            }
        }
        if !required_next.is_empty() {
            let required_next_str = String::from_utf8_lossy(&required_next);
            match_required_next.push(quote! {
                #state => #required_next_str,
            });
        }
    }

    let state_type = if max_state <= u8::MAX as usize {
        quote! { u8 }
    } else if max_state <= u16::MAX as usize {
        quote! { u16 }
    } else if max_state <= u32::MAX as usize {
        quote! { u32 }
    } else if max_state <= u64::MAX as usize {
        quote! { u64 }
    } else {
        quote! { u128 }
    };

    let impl_parser_state = quote! {
        #[derive(Debug, Clone, Copy)]
        struct #parser_state(#state_type);

        impl #parser_state {
            const fn new() -> Self {
                Self(0)
            }
        }
    };

    let parser = format_ident!("{}Parser", ty);
    let impl_parser = quote! {
        struct #parser;

        impl kalosm_sample::CreateParserState for #parser {
            fn create_parser_state(&self) -> <Self as kalosm_sample::Parser>::PartialState {
                #parser_state::new()
            }
        }
        impl kalosm_sample::Parser for #parser {
            type Output = #ty;
            type PartialState = #parser_state;

            fn parse<'a>(
                &self,
                state: &Self::PartialState,
                input: &'a [u8],
            ) -> kalosm_sample::ParseResult<kalosm_sample::ParseStatus<'a, Self::PartialState, Self::Output>> {
                let mut state = *state;
                for (i, byte) in input.iter().enumerate() {
                    match state.0 {
                        #(#parse_states)*
                        _ => return kalosm_sample::ParseResult::Err(kalosm_sample::ParserError::msg("Invalid state")),
                    }
                }
                kalosm_sample::ParseResult::Ok(kalosm_sample::ParseStatus::Incomplete {
                    new_state: state,
                    required_next: std::borrow::Cow::Borrowed(match state.0 {
                        #(#match_required_next)*
                        _ => ""
                    }),
                })
            }
        }
    };

    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                #impl_parser_state
                #impl_parser

                #parser
            }
        }
    }
}

fn unit_enum_schema(data: DataEnum, ty: Ident) -> TokenStream2 {
    let mut variants = Vec::new();
    for variant in data.variants.iter() {
        let variant_name = &variant.ident;
        let literal_string = match unit_parse_literal_name(&variant.attrs, variant_name) {
            Ok(literal_string) => literal_string,
            Err(err) => return err.to_compile_error(),
        };
        variants.push(literal_string);
    }

    let schema = unit_enum_schema_type(variants);

    quote! {
        impl kalosm_sample::Schema for #ty {
            fn schema() -> kalosm_sample::SchemaType {
                #schema
            }
        }
    }
}

fn unit_enum_schema_type(variants: impl IntoIterator<Item = String>) -> proc_macro2::TokenStream {
    let variants = variants.into_iter().map(|variant| {
        let variant = LitStr::new(&variant, variant.span());
        quote! {
            kalosm_sample::SchemaLiteral::String(#variant.to_string())
        }
    });

    let schema = quote! {
        kalosm_sample::EnumSchema::new([#(#variants),*])
    };

    quote! {
        kalosm_sample::SchemaType::Enum(#schema)
    }
}

fn wrap_tuple(ident: &Ident, current: TokenStream2) -> TokenStream2 {
    quote! {
        (#current, #ident)
    }
}

fn parse_rename_attribute(meta: &ParseNestedMeta) -> syn::Result<Option<LitStr>> {
    if meta.path.is_ident("rename") {
        let value = meta
            .value()
            .and_then(|value| value.parse::<syn::LitStr>())?;
        return Ok(Some(value));
    }
    Ok(None)
}

struct FieldsParser {
    fields: Vec<FieldParser>,
}

impl FieldsParser {
    fn new(fields: &[Field]) -> syn::Result<Self> {
        Ok(Self {
            fields: fields
                .iter()
                .map(FieldParser::new)
                .collect::<syn::Result<_>>()?,
        })
    }

    fn parser(&self, construct: TokenStream2) -> syn::Result<TokenStream2> {
        let mut parsers = Vec::new();
        let idents: Vec<_> = self
            .fields
            .iter()
            .map(|f| format_ident!("{}_parser", f.field.ident.as_ref().unwrap().unraw()))
            .collect();
        for (i, (field, parser_ident)) in self.fields.iter().zip(idents.iter()).enumerate() {
            let mut literal_text = String::new();
            if i == 0 {
                literal_text.push_str("{ ");
            } else {
                literal_text.push_str(", ");
            }
            let field_name = &field.name;
            let field_parser = &field.parser;
            literal_text.push_str(&format!("\"{field_name}\": "));
            let literal_text = LitStr::new(&literal_text, field.field.ident.span());

            parsers.push(quote! {
                let #parser_ident = kalosm_sample::ParserExt::ignore_output_then(
                    kalosm_sample::LiteralParser::from(#literal_text),
                    #field_parser
                );
            });
        }

        let mut output_tuple = None;
        for field in self.fields.iter() {
            let name = field.field.ident.as_ref().unwrap();
            match output_tuple {
                Some(current) => {
                    output_tuple = Some(wrap_tuple(name, current));
                }
                None => {
                    output_tuple = Some(name.to_token_stream());
                }
            }
        }

        let mut join_parser: Option<TokenStream2> = None;
        for ident in idents.iter() {
            match &mut join_parser {
                Some(current) => {
                    *current = quote! {
                        kalosm_sample::ParserExt::then(#current, #ident)
                    };
                }
                None => {
                    join_parser = Some(ident.to_token_stream());
                }
            }
        }

        Ok(quote! {
            {
                #(
                    #parsers
                )*

                kalosm_sample::ParserExt::map_output(
                    kalosm_sample::ParserExt::then_literal(
                        #join_parser,
                        r#" }"#
                    ),
                    |#output_tuple| #construct
                )
            }
        })
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        let properties = self.fields.iter().map(|field| field.quote_schema());
        quote! {
            kalosm_sample::JsonObjectSchema::new(
                vec![#(#properties),*]
            )
        }
    }
}

struct FieldParser {
    field: Field,
    parser: Parser,
    name: String,
}

impl FieldParser {
    fn new(field: &Field) -> syn::Result<Self> {
        let mut field_name = field.ident.as_ref().unwrap().unraw().to_string();
        let mut parser: Parser = syn::parse2(field.ty.to_token_stream())?;

        // Look for #[parse(rename = "name")] or #[parse(with = expr)] attributes
        for attr in field.attrs.iter() {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if let Some(value) = parse_rename_attribute(&meta)? {
                        field_name = value.value();
                        Ok(())
                    } else {
                        let attribute_applied = parser.apply_attribute(&meta)?;
                        if !attribute_applied {
                            let mut possible_attributes = vec!["rename"];
                            possible_attributes.extend(parser.possible_attributes());
                            return Err(meta.error(expected_attributes_error(possible_attributes)));
                        }
                        Ok(())
                    }
                })?;
            }
        }

        Ok(Self {
            field: field.clone(),
            parser,
            name: field_name,
        })
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        let schema = self.parser.quote_schema();
        let name = &self.name;
        let description = doc_comment(&self.field.attrs);
        let description = description.map(|description| quote! { .with_description(#description) });
        quote! {
            kalosm_sample::JsonPropertySchema::new(#name.to_string(), #schema)
                .with_required(true)
                #description
        }
    }
}

fn doc_comment(attrs: &[syn::Attribute]) -> Option<String> {
    let mut description = String::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let syn::Meta::NameValue(meta) = &attr.meta else {
            continue;
        };
        if let Ok(lit_str) = syn::parse2::<syn::LitStr>(meta.value.to_token_stream()) {
            if !description.is_empty() {
                description.push('\n');
            }
            let value = lit_str.value();
            let mut borrowed = &*value;
            if borrowed.starts_with(' ') {
                borrowed = &borrowed[1..];
            }
            description.push_str(borrowed);
        }
    }
    (!description.is_empty()).then_some(description)
}

fn expected_attributes_error(
    expected_attributes: impl IntoIterator<Item = &'static str>,
) -> String {
    let mut error_message = String::from("Expected one of the following attributes: ");
    let expected_attributes = expected_attributes.into_iter().collect::<Vec<_>>();
    for (i, attribute) in expected_attributes.iter().enumerate() {
        error_message.push_str(attribute);
        match i.cmp(&(expected_attributes.len().saturating_sub(2))) {
            Ordering::Less => {
                error_message.push_str(", ");
            }
            Ordering::Equal => {
                error_message.push_str(" or ");
            }
            Ordering::Greater => {}
        }
    }
    error_message
}

#[derive(Debug)]
enum ParserType {
    String(StringParserOptions),
    Number(NumberParserOptions),
    Integer(NumberParserOptions),
    Boolean(BoolOptions),
    Custom(proc_macro2::TokenStream),
}

impl Parse for ParserType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Ident) {
            let path = input.parse::<Path>()?;
            if let Ok(string) = StringParserOptions::from_path(&path) {
                return Ok(Self::String(string));
            } else if let Ok(number) = NumberParserOptions::from_path(&path) {
                return Ok(match number.ty {
                    NumberType::F64 | NumberType::F32 => Self::Number(number),
                    _ => Self::Integer(number),
                });
            } else if let Ok(boolean) = BoolOptions::from_path(&path) {
                return Ok(Self::Boolean(boolean));
            }
            Ok(Self::Custom(path.to_token_stream()))
        } else {
            Ok(Self::Custom(input.parse()?))
        }
    }
}

#[test]
fn type_parses() {
    assert!(matches!(
        syn::parse2::<ParserType>(quote! { String }).unwrap(),
        ParserType::String(_)
    ));
    assert!(matches!(
        syn::parse2::<ParserType>(quote! { std::string::String }).unwrap(),
        ParserType::String(_)
    ));
    assert!(matches!(
        syn::parse2::<ParserType>(quote! { i32 }).unwrap(),
        ParserType::Integer(_)
    ));
    assert!(matches!(
        syn::parse2::<ParserType>(quote! { f32 }).unwrap(),
        ParserType::Number(_)
    ));
}

#[derive(Debug)]
struct Parser {
    ty: ParserType,
    with: Option<proc_macro2::TokenStream>,
    schema: Option<proc_macro2::TokenStream>,
}

impl Parse for Parser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            ty: input.parse()?,
            with: None,
            schema: None,
        })
    }
}

impl Parser {
    fn apply_attribute(&mut self, input: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
        if input.path.is_ident("with") {
            self.with = Some(input.value()?.parse()?);
            Ok(true)
        } else if input.path.is_ident("schema") {
            self.schema = Some(input.value()?.parse()?);
            Ok(true)
        } else {
            match &mut self.ty {
                ParserType::String(options) => options.apply_attribute(input),
                ParserType::Number(options) => options.apply_attribute(input),
                ParserType::Integer(options) => options.apply_attribute(input),
                ParserType::Boolean(options) => options.apply_attribute(input),
                ParserType::Custom(_) => Ok(false),
            }
        }
    }

    fn possible_attributes(&self) -> Vec<&'static str> {
        let mut attributes = vec!["with", "schema"];
        match &self.ty {
            ParserType::String(_) => attributes.extend(StringParserOptions::ATTRIBUTES),
            ParserType::Integer(_) | ParserType::Number(_) => {
                attributes.extend(NumberParserOptions::ATTRIBUTES)
            }
            ParserType::Boolean(_) => attributes.extend(BoolOptions::ATTRIBUTES),
            _ => {}
        }
        attributes
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        if let Some(schema) = &self.schema {
            return schema.clone();
        }

        match &self.ty {
            ParserType::String(options) => {
                let schema = options.quote_schema();
                quote! {
                    kalosm_sample::SchemaType::String(#schema)
                }
            }
            ParserType::Number(options) => {
                let schema = options.quote_schema();
                quote! {
                    kalosm_sample::SchemaType::Number(#schema)
                }
            }
            ParserType::Integer(options) => {
                let schema = options.quote_schema();
                quote! {
                    kalosm_sample::SchemaType::Integer(#schema)
                }
            }
            ParserType::Boolean(options) => {
                let schema = options.quote_schema();
                quote! {
                    kalosm_sample::SchemaType::Boolean(#schema)
                }
            }
            ParserType::Custom(ty) => {
                quote_spanned! {
                    ty.span() =>
                    <#ty as kalosm_sample::Schema>::schema()
                }
            }
        }
    }
}

impl ToTokens for Parser {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        if let Some(with) = &self.with {
            with.to_tokens(tokens);
            return;
        }

        tokens.extend(match &self.ty {
            ParserType::String(options) => {
                quote! {
                    #options
                }
            }
            ParserType::Integer(options) => {
                quote! {
                    #options
                }
            }
            ParserType::Number(options) => {
                quote! {
                    #options
                }
            }
            ParserType::Boolean(options) => {
                quote! {
                    #options
                }
            }
            ParserType::Custom(ty) => {
                quote! {
                    <#ty as kalosm_sample::Parse>::new_parser()
                }
            }
        })
    }
}

// Strings accept these attributes:
// - #[parse(character_filter = |c| ...)]
// - #[parse(len = 1..=10)]
// - #[parse(pattern = "a+")]
struct StringParserOptions {
    path: Path,
    character_filter: Option<proc_macro2::TokenStream>,
    len: Option<proc_macro2::TokenStream>,
    pattern: Option<LitStr>,
}

impl Debug for StringParserOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringParserOptions")
            .field("character_filter", &self.character_filter)
            .field("len", &self.len)
            .field("pattern", &self.pattern.as_ref().map(|p| p.value()))
            .finish()
    }
}

impl Parse for StringParserOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path = input.parse()?;
        Self::from_path(&path)
    }
}

impl StringParserOptions {
    const ATTRIBUTES: &'static [&'static str] = &["character_filter", "len", "pattern"];

    fn apply_attribute(&mut self, input: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
        if input.path.is_ident("character_filter") {
            self.character_filter = Some(input.value()?.parse()?);
            Ok(true)
        } else if input.path.is_ident("len") {
            self.len = Some(input.value()?.parse()?);
            Ok(true)
        } else if input.path.is_ident("pattern") {
            self.pattern = Some(input.value()?.parse()?);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn from_path(path: &Path) -> syn::Result<Self> {
        if !is_string(path) {
            return Err(syn::Error::new(path.span(), "Expected a string type"));
        }
        Ok(Self {
            path: path.clone(),
            character_filter: None,
            len: None,
            pattern: None,
        })
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        let len = self.len.as_ref().map(|len| {
            quote_spanned! {
                len.span() =>
                .with_length(#len)
            }
        });
        let pattern = self.pattern.as_ref().map(|pattern| {
            quote_spanned! {
                pattern.span() =>
                .with_pattern(#pattern)
            }
        });
        let quote = quote_spanned! {
            self.path.span() =>
            kalosm_sample::StringSchema::new()
            #len
            #pattern
        };
        quote
    }
}

impl ToTokens for StringParserOptions {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        if let Some(pattern) = &self.pattern {
            let pattern_str = pattern.value();
            let mut pattern_str = pattern_str.as_str();
            // Strip the ^ and $ from the pattern if they are present. The parser will automatically parse the whole contents
            if let Some(new_pattern_str) = pattern_str.strip_prefix("^") {
                pattern_str = new_pattern_str;
            }
            if !pattern_str.ends_with("\\$") {
                if let Some(new_pattern_str) = pattern_str.strip_suffix('$') {
                    pattern_str = new_pattern_str;
                }
            }
            let pattern = LitStr::new(&format!(r#""{pattern_str}""#), pattern.span());
            let quote = quote_spanned! {
                pattern.span() =>
                kalosm_sample::ParserExt::map_output(
                    kalosm_sample::RegexParser::new(#pattern)
                        .unwrap(),
                    // Trim the quotes
                    |string| string[1..string.len() - 1].to_string()
                )
            };
            tokens.extend(quote);
            return;
        }

        let character_filter = self.character_filter.as_ref().map(|filter| {
            let quote = quote_spanned! {
                filter.span() =>
                .with_allowed_characters(#filter)
            };
            quote
        });
        let len = self
            .len
            .as_ref()
            .map(|len| len.to_token_stream())
            .unwrap_or_else(|| {
                quote! {
                    0..=usize::MAX
                }
            });
        let quote = quote_spanned! {
            self.path.span() =>
            kalosm_sample::StringParser::new(#len)
            #character_filter
        };
        tokens.extend(quote);
    }
}

fn is_string(ty: &syn::Path) -> bool {
    let string_path = syn::parse_quote!(::std::string::String);
    is_path_type(ty, &string_path)
}

#[derive(Debug)]
enum NumberType {
    F64,
    F32,
    I128,
    I64,
    I32,
    I16,
    I8,
    Isize,
    U128,
    U64,
    U32,
    U16,
    U8,
    Usize,
}

impl Parse for NumberType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ty = input.parse()?;

        Self::from_path(&ty)
    }
}

impl NumberType {
    fn from_path(ty: &Path) -> syn::Result<Self> {
        let f64_path = syn::parse_quote!(::std::primitive::f64);
        if is_path_type(ty, &f64_path) {
            return Ok(Self::F64);
        }

        let f32_path = syn::parse_quote!(::std::primitive::f32);
        if is_path_type(ty, &f32_path) {
            return Ok(Self::F32);
        }

        let i128_path = syn::parse_quote!(::std::primitive::i128);
        if is_path_type(ty, &i128_path) {
            return Ok(Self::I128);
        }

        let i64_path = syn::parse_quote!(::std::primitive::i64);
        if is_path_type(ty, &i64_path) {
            return Ok(Self::I64);
        }

        let i32_path = syn::parse_quote!(::std::primitive::i32);
        if is_path_type(ty, &i32_path) {
            return Ok(Self::I32);
        }

        let i16_path = syn::parse_quote!(::std::primitive::i16);
        if is_path_type(ty, &i16_path) {
            return Ok(Self::I16);
        }

        let i8_path = syn::parse_quote!(::std::primitive::i8);
        if is_path_type(ty, &i8_path) {
            return Ok(Self::I8);
        }

        let isize_path = syn::parse_quote!(::std::primitive::isize);
        if is_path_type(ty, &isize_path) {
            return Ok(Self::Isize);
        }

        let u128_path = syn::parse_quote!(::std::primitive::u128);
        if is_path_type(ty, &u128_path) {
            return Ok(Self::U128);
        }

        let u64_path = syn::parse_quote!(::std::primitive::u64);
        if is_path_type(ty, &u64_path) {
            return Ok(Self::U64);
        }

        let u32_path = syn::parse_quote!(::std::primitive::u32);
        if is_path_type(ty, &u32_path) {
            return Ok(Self::U32);
        }

        let u16_path = syn::parse_quote!(::std::primitive::u16);
        if is_path_type(ty, &u16_path) {
            return Ok(Self::U16);
        }

        let u8_path = syn::parse_quote!(::std::primitive::u8);
        if is_path_type(ty, &u8_path) {
            return Ok(Self::U8);
        }

        let usize_path = syn::parse_quote!(::std::primitive::usize);
        if is_path_type(ty, &usize_path) {
            return Ok(Self::Usize);
        }

        Err(syn::Error::new(ty.span(), "Expected a number type"))
    }
}

impl ToTokens for NumberType {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let quote = match self {
            Self::F64 => quote! {kalosm_sample::F64Parser::new()},
            Self::F32 => quote! {kalosm_sample::F32Parser::new()},
            Self::I128 => quote! {kalosm_sample::I128Parser::new()},
            Self::I64 => quote! {kalosm_sample::I64Parser::new()},
            Self::I32 => quote! {kalosm_sample::I32Parser::new()},
            Self::I16 => quote! {kalosm_sample::I16Parser::new()},
            Self::I8 => quote! {kalosm_sample::I8Parser::new()},
            Self::Isize => quote! {kalosm_sample::IsizeParser::new()},
            Self::U128 => quote! {kalosm_sample::U128Parser::new()},
            Self::U64 => quote! {kalosm_sample::U64Parser::new()},
            Self::U32 => quote! {kalosm_sample::U32Parser::new()},
            Self::U16 => quote! {kalosm_sample::U16Parser::new()},
            Self::U8 => quote! {kalosm_sample::U8Parser::new()},
            Self::Usize => quote! {kalosm_sample::UsizeParser::new()},
        };

        tokens.extend(quote);
    }
}

// Numbers accept these attributes:
// - #[parse(range = 0.0..=100.0)]
struct NumberParserOptions {
    path: Path,
    ty: NumberType,
    range: Option<proc_macro2::TokenStream>,
}

impl Debug for NumberParserOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumberParserOptions")
            .field("ty", &self.ty)
            .field("range", &self.range)
            .finish()
    }
}

impl Parse for NumberParserOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path = input.parse::<Path>()?;

        Self::from_path(&path)
    }
}

impl NumberParserOptions {
    fn from_path(path: &Path) -> syn::Result<Self> {
        let ty = NumberType::from_path(path)?;
        Ok(Self {
            path: path.clone(),
            ty,
            range: None,
        })
    }
}

impl ToTokens for NumberParserOptions {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let range = self.range.as_ref().map(|range| {
            let quote = quote_spanned! {
                range.span() =>
                .with_range(#range)
            };
            quote
        });
        let ty = &self.ty;
        let quote = quote_spanned! {
            self.path.span() =>
            #ty
            #range
        };
        tokens.extend(quote);
    }
}

impl NumberParserOptions {
    const ATTRIBUTES: &'static [&'static str] = &["range"];

    fn apply_attribute(&mut self, input: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
        if input.path.is_ident("range") {
            self.range = Some(input.value()?.parse()?);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        match self.ty {
            NumberType::F64 | NumberType::F32 => {
                let range = self.range.as_ref().map(|range| {
                    quote_spanned! {
                        range.span() =>
                            .with_range({
                                let range = #range;
                                let start = range.start() as f64;
                                let end = range.end() as f64;
                                start..=end
                            })
                    }
                });
                quote_spanned! {
                    self.path.span() =>
                    kalosm_sample::NumberSchema::new()
                    #range
                }
            }
            _ => quote_spanned! {
                self.path.span() =>
                kalosm_sample::IntegerSchema::new()
            },
        }
    }
}

struct BoolOptions {
    path: Path,
}

impl Debug for BoolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoolOptions").finish()
    }
}

impl BoolOptions {
    const ATTRIBUTES: &'static [&'static str] = &[];

    fn apply_attribute(&mut self, _input: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
        Ok(false)
    }

    fn from_path(path: &Path) -> syn::Result<Self> {
        if !is_bool(path) {
            return Err(syn::Error::new(path.span(), "Expected a boolean type"));
        }
        Ok(Self { path: path.clone() })
    }

    fn quote_schema(&self) -> proc_macro2::TokenStream {
        quote_spanned! {
            self.path.span() =>
            kalosm_sample::BooleanSchema::new()
        }
    }
}

impl Parse for BoolOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path = input.parse()?;
        Self::from_path(&path)
    }
}

impl ToTokens for BoolOptions {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let quote = quote! {
            <bool as kalosm_sample::Parse>::new_parser()
        };
        tokens.extend(quote);
    }
}

fn is_bool(ty: &syn::Path) -> bool {
    let bool_path = syn::parse_quote!(::std::bool);
    is_path_type(ty, &bool_path)
}

// Check if the last type segment matches a value
fn is_path_type(path: &syn::Path, match_path: &TypePath) -> bool {
    let mut path_segments = path.segments.iter().rev();
    let mut match_path_segments = match_path.path.segments.iter().rev();
    loop {
        match (path_segments.next(), match_path_segments.next()) {
            (Some(first), Some(second)) => {
                if first.ident != second.ident {
                    return false;
                }
            }
            (None, None) => return true,
            (Some(_), None) => return false,
            (None, Some(_)) => return true,
        }
    }
}
