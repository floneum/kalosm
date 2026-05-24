use kalosm::language::*;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn main() {
    pollster::block_on(async {
        let real_visited = Arc::new(AtomicUsize::new(0));
        Page::crawl(
            Url::parse("https://dioxuslabs.com/learn/0.5/").unwrap(),
            BrowserMode::Static,
            move |page: Page| {
                let real_visited = real_visited.clone();
                Box::pin(async move {
                    let visited = real_visited.fetch_add(1, Ordering::SeqCst);

                    if page.url().domain() != Some("dioxuslabs.com") {
                        return CrawlFeedback::follow_none();
                    }
                    let path_prefix = "/learn/0.5/";
                    if !page.url().path().starts_with(path_prefix) {
                        return CrawlFeedback::follow_none();
                    }

                    let Ok(mut document) = page.html().await else {
                        return CrawlFeedback::follow_none();
                    };

                    let original_length = document.html().len();

                    let mut simplifier = HtmlSimplifier::default();
                    simplifier.simplify(&mut document);
                    let simplified = document.html();
                    let simplified_length = simplified.len();
                    let percentage_decrease =
                        (original_length - simplified_length) as f32 / original_length as f32;
                    println!(
                        "simplifing {} -{:.3}% from {:?} to {:?}",
                        page.url(),
                        percentage_decrease,
                        original_length,
                        simplified_length
                    );

                    // write the page to disk
                    let _ = std::fs::create_dir_all("scraped");
                    if let Ok(mut file) = std::fs::File::create(format!("scraped/{visited}.html")) {
                        _ = file.write_all(simplified.as_bytes());
                    }

                    CrawlFeedback::follow_all()
                }) as Pin<Box<dyn Future<Output = CrawlFeedback>>>
            },
        )
        .await;
    });
}
