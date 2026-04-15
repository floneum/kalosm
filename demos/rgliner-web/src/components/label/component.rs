use dioxus::prelude::*;
use dioxus_primitives::label::{self, LabelProps};

#[component]
pub fn Label(props: LabelProps) -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: asset!("./style.css") }
        label::Label {
            class: "label",
            html_for: props.html_for,
            attributes: props.attributes,
            {props.children}
        }
    }
}
