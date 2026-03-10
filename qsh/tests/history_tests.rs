use qsh::history::History;
use tempfile::NamedTempFile;

#[test]
fn test_history_basic() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();

    let history = History::open_at(path).expect("Failed to open history");

    history
        .add_message("user", "Hello", None)
        .expect("Failed to add message");
    history
        .add_message("assistant", "Hi there", None)
        .expect("Failed to add message");

    let messages = history
        .get_recent_messages(10)
        .expect("Failed to get messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].0, "user");
    assert_eq!(messages[0].1, "Hello");
    assert_eq!(messages[1].0, "assistant");
    assert_eq!(messages[1].1, "Hi there");
}

#[test]
fn test_history_limit() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();

    let history = History::open_at(path).expect("Failed to open history");

    for i in 0..10 {
        history
            .add_message("user", &format!("Message {}", i), None)
            .unwrap();
    }

    let messages = history
        .get_recent_messages(5)
        .expect("Failed to get messages");
    assert_eq!(messages.len(), 5);
    // get_recent_messages reverses them back to chronological, so we should have messages 5, 6, 7, 8, 9
    assert_eq!(messages[0].1, "Message 5");
    assert_eq!(messages[4].1, "Message 9");
}

#[test]
fn test_history_clear() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();

    let history = History::open_at(path).expect("Failed to open history");

    history.add_message("user", "test", None).unwrap();
    history.clear().expect("Failed to clear history");

    let messages = history
        .get_recent_messages(10)
        .expect("Failed to get messages");
    assert!(messages.is_empty());
}
