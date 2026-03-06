// ─────────────────────────────────────────────────────────────────────
// test_ring_buffer.cpp — Validation of SPSC queue Drop-Head backpressure
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <atomic>
#include <cstdint>

// Test the core backpressure mechanism directly
// (same logic as ZmqReceiver::run but without ZMQ dependency)

struct TestMsg {
    uint64_t seq;
    std::string data;
};

TEST(RingBuffer, BasicEnqueueDequeue) {
    moodycamel::ReaderWriterQueue<TestMsg> queue{100};
    
    queue.try_enqueue(TestMsg{1, "hello"});
    
    TestMsg out;
    ASSERT_TRUE(queue.try_dequeue(out));
    EXPECT_EQ(out.seq, 1);
    EXPECT_EQ(out.data, "hello");
}

TEST(RingBuffer, BoundedCapacityOverflow) {
    constexpr size_t CAPACITY = 10;
    moodycamel::ReaderWriterQueue<TestMsg> queue{CAPACITY};
    std::atomic<bool> data_loss_flag{false};

    // Fill the queue
    for (size_t i = 0; i < CAPACITY; ++i) {
        ASSERT_TRUE(queue.try_enqueue(TestMsg{i, "msg"}));
    }

    // Next enqueue should fail (queue full)
    TestMsg overflow_msg{999, "overflow"};
    ASSERT_FALSE(queue.try_enqueue(overflow_msg));
}

TEST(RingBuffer, DropHeadPolicy) {
    constexpr size_t CAPACITY = 5;
    moodycamel::ReaderWriterQueue<TestMsg> queue{CAPACITY};
    std::atomic<bool> data_loss_flag{false};

    // Fill queue with messages 0..4
    for (uint64_t i = 0; i < CAPACITY; ++i) {
        queue.try_enqueue(TestMsg{i, "old"});
    }

    // Apply Drop-Head: discard oldest, insert new
    TestMsg new_msg{100, "new"};
    if (!queue.try_enqueue(new_msg)) {
        TestMsg dropped;
        queue.try_dequeue(dropped);  // Drop oldest (seq=0)
        EXPECT_EQ(dropped.seq, 0);
        
        ASSERT_TRUE(queue.try_enqueue(new_msg));
        data_loss_flag.store(true, std::memory_order_relaxed);
    }

    // Verify data loss flag was set
    EXPECT_TRUE(data_loss_flag.load());

    // Drain remaining: should be [1, 2, 3, 4, 100]
    std::vector<uint64_t> seqs;
    TestMsg out;
    while (queue.try_dequeue(out)) {
        seqs.push_back(out.seq);
    }

    ASSERT_EQ(seqs.size(), CAPACITY);
    EXPECT_EQ(seqs.front(), 1);  // Oldest surviving
    EXPECT_EQ(seqs.back(), 100); // Newest inserted
}

TEST(RingBuffer, EmptyDequeueReturns_false) {
    moodycamel::ReaderWriterQueue<TestMsg> queue{10};
    TestMsg out;
    EXPECT_FALSE(queue.try_dequeue(out));
}

TEST(RingBuffer, NoMemoryLeakAfterOverflow) {
    constexpr size_t CAPACITY = 100;
    moodycamel::ReaderWriterQueue<TestMsg> queue{CAPACITY};

    // Simulate 1M messages with continuous Drop-Head
    for (uint64_t i = 0; i < 1'000'000; ++i) {
        TestMsg msg{i, std::string(64, 'x')};  // 64-byte payload
        if (!queue.try_enqueue(std::move(msg))) {
            TestMsg dropped;
            queue.try_dequeue(dropped);
            queue.try_enqueue(TestMsg{i, std::string(64, 'x')});
        }
    }

    // Drain — should not crash or leak
    TestMsg out;
    size_t count = 0;
    while (queue.try_dequeue(out)) ++count;
    EXPECT_GT(count, 0u);
    EXPECT_LE(count, CAPACITY);
}
