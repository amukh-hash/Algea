// ─────────────────────────────────────────────────────────────────────
// test_alert_dag.cpp — Validation of DAG root-cause analysis
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "engine/AlertDag.h"

using algae::engine::AlertDag;

class AlertDagTest : public ::testing::Test {
protected:
    AlertDag dag;
};

TEST_F(AlertDagTest, RootCauseNotInhibited) {
    dag.processAlert("root1", "", 3, "Correlation breach", "risk_engine", 0);
    
    EXPECT_EQ(dag.activeAlertCount(), 1);
    EXPECT_EQ(dag.inhibitedCount(), 0);
    
    auto visible = dag.visibleAlerts();
    ASSERT_EQ(visible.size(), 1);
    EXPECT_TRUE(visible[0].toMap()["isRootCause"].toBool());
}

TEST_F(AlertDagTest, SymptomIsInhibitedWhenRootCauseActive) {
    // Root cause fires first
    dag.processAlert("root1", "", 3, "Correlation > 2.0", "risk_engine", 100);
    
    // Symptom references the root cause
    dag.processAlert("symptom1", "root1", 2, "Low volume execution", "executor", 200);
    
    EXPECT_EQ(dag.activeAlertCount(), 2);
    EXPECT_EQ(dag.inhibitedCount(), 1);
    
    // Only root cause should be visible
    auto visible = dag.visibleAlerts();
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].toMap()["id"].toString(), "root1");
}

TEST_F(AlertDagTest, SymptomUnInhibitedWhenRootCauseCleared) {
    dag.processAlert("root1", "", 3, "Root cause", "risk_engine", 100);
    dag.processAlert("symptom1", "root1", 2, "Symptom", "executor", 200);
    
    EXPECT_EQ(dag.inhibitedCount(), 1);
    
    // Clear root cause — symptom should become visible
    dag.clearAlert("root1");
    
    EXPECT_EQ(dag.activeAlertCount(), 1);
    EXPECT_EQ(dag.inhibitedCount(), 0);
    
    auto visible = dag.visibleAlerts();
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].toMap()["id"].toString(), "symptom1");
}

TEST_F(AlertDagTest, MultipleSymptomsSuppressed) {
    dag.processAlert("root1", "", 3, "Root cause", "risk_engine", 100);
    dag.processAlert("sym1", "root1", 2, "Symptom 1", "executor", 200);
    dag.processAlert("sym2", "root1", 1, "Symptom 2", "logger", 300);
    dag.processAlert("sym3", "root1", 1, "Symptom 3", "monitor", 400);
    
    EXPECT_EQ(dag.activeAlertCount(), 4);
    EXPECT_EQ(dag.inhibitedCount(), 3);
    
    auto visible = dag.visibleAlerts();
    ASSERT_EQ(visible.size(), 1);
}

TEST_F(AlertDagTest, IndependentAlertNotInhibited) {
    dag.processAlert("root1", "", 3, "Root cause", "risk_engine", 100);
    dag.processAlert("independent1", "", 2, "Unrelated alert", "other", 200);
    
    EXPECT_EQ(dag.activeAlertCount(), 2);
    EXPECT_EQ(dag.inhibitedCount(), 0);
    
    auto visible = dag.visibleAlerts();
    EXPECT_EQ(visible.size(), 2);
}

TEST_F(AlertDagTest, ClearAllResetsState) {
    dag.processAlert("root1", "", 3, "Root", "risk", 100);
    dag.processAlert("sym1", "root1", 2, "Sym", "exec", 200);
    
    dag.clearAll();
    
    EXPECT_EQ(dag.activeAlertCount(), 0);
    EXPECT_EQ(dag.inhibitedCount(), 0);
    EXPECT_TRUE(dag.visibleAlerts().isEmpty());
}
