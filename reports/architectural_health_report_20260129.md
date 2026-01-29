# HydraNet Architectural Health & Robustness Report
**Date:** 29-01-2026
**Subject:** Diagnostic findings from the End-to-End Flight Test implementation.

---

## Section 1: The "Flight Test" Diagnostic Summary

### 1.1. Context: What just happened?
We attempted to create a "Smoke Test" (`tests/test_end_to_end_smoke.py`) that would prove the system can run a mission from start to finish. This test was unexpectedly difficult to write. In software engineering, **difficulty in testing is a direct symptom of architectural entanglement.**

### 1.2. The Core Discovery
The test revealed that HydraNet is currently **tightly coupled** to the internal mechanics of the `views-pipeline-core`. We are not just using the library; we are trapped inside its assumptions.

### 1.3. Key "Pain Points" Found
*   **Invisible Requirements:** The system requires specific `.txt` log files to exist in specific places, but the code never asks for them explicitly. They are "side-effect" dependencies.
*   **The "God Class" Problem:** Because we inherit from `ForecastingModelManager`, we inherit all its "noise." We cannot test HydraNet without also testing the entire Pipeline Core's initialization logic.
*   **Managed State Confusion:** The way configurations are handled makes it hard to predict what the system knows at any given second.

### 1.4. The "HFC" (HydraNet Flight Check) Shield
Despite the difficulty, we now have a **Shield**. The new smoke test physically constructs a tiny version of the world.
*   **If the Core changes, the test fails.**
*   **If a file is missing, the test fails.**
*   **No more silent "Lying Mocks."**

---

### **Actionable Summary for Logical Clarity**
*   **The Problem:** The system expects a specific folder structure that isn't written down in the functions.
*   **The Fix we implemented:** We now "Mirror" the folder structure perfectly before starting.
*   **The Lesson:** We need to move from **Inheritance** (being the core) to **Composition** (using the core).

---

## Section 2: The "Invisible Filesystem Contract" Deep-Dive

### 2.1. The "Magic" Folders
The `views-pipeline-core` library operates on an implicit assumption: **The location of the data is the location of the metadata.** 

*   **The Assumption:** If I am reading a Parquet file from `/data/raw/`, I should also find a file called `/data/raw/calibration_data_fetch_log.txt`.
*   **The Failure:** When we tried to be "clean" by pointing the system to an isolated augmented file, the system looked for the companion log in our empty temporary folder and crashed (`FileNotFoundError`).

### 2.2. Why this is Brittle (Opacity)
This contract is "invisible" because:
1.  It is not passed as a variable to the function.
2.  It is not documented in the docstrings.
3.  It is hardcoded as string-concatenation inside the core library.

### 2.3. The "Shadow Directory" Solution
To fix this without rewriting the core library, we implemented **Environment Mirroring**:
*   We create a temporary `shadow` directory.
*   We put our "Better Data" (augmented Parquet) in it.
*   We **Symlink** the original log files into it.
*   The Core is "tricked" into seeing a complete, valid environment.

---

### **Proposed GitHub Ticket: #001 - Decouple Metadata from Data Paths**
**Description:** Refactor the `ForecastingModelManager` to accept an explicit `metadata_log_path` instead of deriving it from the `data_raw` directory.
**Goal:** Allow models to provide augmented data without having to "mirror" the entire filesystem.
**Clarity:** Stop the system from guessing where files are.

---

### **Actionable Summary for Logical Clarity**
*   **Current State:** The system is like a person who refuses to eat a sandwich unless it's served on a specific blue plate in a specific room.
*   **Our Solution:** We bought a matching blue plate and moved it to the room the system likes.
*   **Desired State:** The system should just eat the sandwich regardless of the plate.

---

## Section 3: The "God Class" Inheritance Problem

### 3.1. The Forced Marriage
`HydranetManager` inherits from `ForecastingModelManager`. This inheritance is **forced**, not voluntary.
*   **Initialization Noise:** In its `__init__`, the base class automatically scans the disk for configs and artifacts. If it doesn't find them, it crashes.
*   **The Testing Nightmare:** You cannot test a "Pure HydraNet Function" without setting up a fake VIEWS project on disk first.
*   **Tight Coupling:** Any change to how the base class initializes (e.g., adding a new required file) will break `HydranetManager` and all its tests.

---

### **Proposed GitHub Ticket: #002 - From Inheritance to Composition**
**Description:** Stop forcing model managers to inherit from `ForecastingModelManager`. Move shared logic (file saving, logging) into a set of **Services** or **Mixins**.
**Goal:** Allow `HydranetManager` to exist as a standalone entity that *uses* pipeline tools rather than *being* the pipeline.
**Clarity:** Make the model independent of the pipeline infrastructure.

---

### **Actionable Summary for Logical Clarity**
*   **The Problem:** HydraNet is currently a "conjoined twin" with the Pipeline Core. We can't move one without hurting the other.
*   **The Fix we want:** We want to perform "separation surgery" so that HydraNet can live in its own house and just "call" the Pipeline when it needs something.

---

## Section 4: The "Managed State Trap"

### 4.1. Side-Effects in Setters
The `configs` property in the Core library is not a simple dictionary. It has a **Setter with Side-Effects**.
*   **The Issue:** When you do `manager.configs = new_config`, the library doesn't just store the data; it tries to update an internal `_config_manager` object.
*   **The Failure:** If that internal object isn't perfectly set up, the assignment crashes.
*   **Non-Idiomatic Python:** In Python, we expect setters to be "cheap" and "predictable." This one is "expensive" and "surprising."

### 4.2. State Inconsistency
The manager maintains two nearly identical dictionaries: `self.config` and `self.configs`. 
*   **The Confusion:** It is never clear which one is the "Source of Truth." 
*   **The Fragility:** We frequently had to manually update both to ensure the system didn't revert to old values during the evaluation loop.

---

### **Proposed GitHub Ticket: #003 - Unified, Immutable Configuration**
**Description:** Replace the managed `@property` setters with a standard `ImmutableConfig` object (using Pydantic).
**Goal:** Ensure that once a configuration is set, it cannot be changed silently by the base class.
**Clarity:** One source of truth, no side effects.

---

### **Actionable Summary for Logical Clarity**
*   **The Problem:** The system's memory is unreliable. You tell it one thing, and it "forgets" or "changes" it because of a hidden rule.
*   **The Fix we want:** We want the system to have a **Solid Memory**. If I write something down, it stays written down exactly as I intended.

---

## Section 5: The "Lying Mocks" Lesson & Final Conclusion

### 5.1. Why 107 Tests Passed while the Pipeline Failed
We fell into the **Mocking Paradox**: We tested our *understanding* of the Pipeline Core, not the *actual* Pipeline Core.
*   **The Error:** Our mocks assumed that `super()._execute_model_evaluation()` was a "black box" that only needed a Parquet file.
*   **The Reality:** The real code was a "transparent box" that reached out and touched the filesystem for log files we didn't even know existed.
*   **The Lesson:** Mocks are only as good as the developer's knowledge. In a complex, coupled system, **Smoke Tests (HFC)** are the only source of truth.

### 5.2. Final Conclusion: The Path Forward
HydraNet is now stable, but it is **"Over-Governed"** by the Pipeline Core. We have successfully mitigated this by:
1.  **Strict Typing:** Pydantic configs prevent the "forgotten key" crashes.
2.  **Explicit Data Flow:** We no longer lie to the Python interpreter; we physically prepare the data the system wants.
3.  **The Flight Shield:** We have a test that physically proves the system can "Fly" before we launch a 100-sample mission.

### **Immediate Next Steps:**
1.  Review and open the 3 proposed GitHub tickets in the `views-pipeline-core` repository.
2.  Maintain the `tests/test_end_to_end_smoke.py` as the **Highest Priority** test. If it breaks, stop everything.

---

### **Actionable Summary for Logical Clarity**
*   **The Lesson:** Logic is easy; integration is hard. We were right about the math, but wrong about the plumbing.
*   **The State of the Union:** The plumbing is now fixed with "Environment Mirroring," but it's a workaround. We need to fix the actual pipes (the Core library).
*   **Final Status:** **READY FOR MISSION.**