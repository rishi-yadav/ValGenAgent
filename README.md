
#  ValGenAgent

ValGenAgent is a **RAG-powered, AI agents based pipeline** for automated:
- Test plan generation (in doc/json format).
- Test case generation (in .py/.cpp/.c/.s etc) using generated test plan or user provided test plan.
- Execution of test cases on target platform (HW/Simulator)

It simplifies software validation workflows by generating test plans (from feature input files), generating
the test cases and, executing tests cases on target device. The pipeline uses existing source code, documents,
public URLs to create a vector index which provides context to LLM (GPT-4o) for better code genetion.
It provides flexibility to user to selectively use all three or single execution flow.

Note: The pipeline uses Azure deployed models for embedding and inference hence it is completely safe to use even with your proprietary code.

Please check the following resources for iGPT access:
- Migration docs from igpt to azure: https://ryan.intel.com/Learning/AI/#Transition
- Azure Docs: https://learn.microsoft.com/en-us/azure/ai-foundry/what-is-azure-ai-foundry

LlamaIndex and autogen are used for pipeline creation.

---

##  Features

-  RAG-Based Test Plan Generation
-  RAG-Based Test Case Generation
-  Code Review is done by Agent
-  Test Case Execution on target device (HW/Simulator) by Agent
-  Test reports are generated with full details
-  Supports source code (c/cpp/python/assembly) + Documents (docx/pptx/pdf etc.) + Pulic URLs as inputs for richer context
-  Langugae aware and Hierarchical parser
-  Context-aware retrieval Pipeline
-  Context can be given as RAG or Files in case of files we can use the flag --add_context_dir

---

End-to-End pipeline

![Alt text](./E2E-pipeline.jpg "End-to-End pipeline")

---

Testcase generation workflow

![Alt text](./workflow-arch.jpg "Workflow Architecture")


---

##  Input Support

ValGenAgent can consume source codes (c/cpp/python/assembly), documents (docx/pptx/pdf etc.) files,
and public URLs (un-restricted access) as inputs for richer context while creating test plan and test codes.

###  It supports language aware parser for the following coding languages

- `C`
- `C++`
- `Python`
- `Assembly`
- Other languages are parsed using a **HierarchicalNodeParser**. More language support is in progress.

###  Supported file formats

ValGenAgent uses [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/) from LlamaIndex, and supports:

- `.txt`, `.md`, `.py`, `.c`, `.cpp`, `.s`, `.json`, `.docx`, `.xlsx`, `.pptx`, `pdf`, etc.
- Unsupported file will be treated as plane text

---
## Where to keep inputs (codes/docs/public_urls etc.) ?
All inputs should be kept in this `input_dirs` directory as follows:

```
input_dirs/
├── code/              # Source code files (C, C++, Python, etc.)
├── docs/              # Supporting documents (e.g., .txt, .md, .pdf)
└── public_urls.txt    # Public URLs for RAG context
```

##  Output Directory Structure

All outputs (plans, scripts, logs) are saved under user provided directory name ex: `test_results/` as follows:

```
test_results/
├── test_plan.docx
├── test_plan.json
│── test_results.xml
│── test_operations.py
│── test_chat_log.txt
└── function_test_results.xlsx
```

---

##  Command-Line Usage

Following are the steps to use it:

###  Initial Setup

1. **Prepare Target Hardware/Simulator env**
   - If you are using Gaudi, ensure you have Gaudi container up and running.
   - For any other system, connect to the target machine.

2. **Clone the Repository**
   ```bash
   git clone https://github.com/pramodkumar-habanalabs/ValGenAgent.git
   cd ValGenAgent
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
> **Note:** If required make a python environment.
---

###  Prepare input directory

Put your code, docs, and public urls inside the `input_dirs`. Alternatively, create a softlink accordingly.

If using --add_context_dir, keep the files in a directory that can be given to the agent as a context.

---

###  Create feature input file in .json format
```
This is the prompt/instruction file used to generate test cases.
    The feature input file should contain:
    {
      "name": "Feature Name",
      "description": "Detailed feature description..."
    }

Please refer feature_description/collective_feature.json
```

If using test_plan you can give in the following format.

```json
{
  "test_plan": "cutlass_flash_attention_decode",
  "tests": {
    "test_category": "cutlass_flash_attention_decode",
    "implementation_file": "xe_flash_decode_generated.cpp",
    "test_cases": [
      {
        "test_id": "REQ_001",
        "description": "Write 5 new tests cases for flash attention decode that is for different LLM configurations eg: gpt,llama etc. you need to overload existing TestFlashDecodeAll function for better quality tests."
      }
    ]
  }
}
```
---

###  Run the Agent

```bash
python test_runner.py --feature_input_file <path_to_input_file> --output_dir test_results prompt_path path/to/system_prompts --execute_python
```

---

###  other example command

```bash
# Just generate test cases - don't run them
python test_runner.py --feature_input_file input_file/collective_feature.json --prompt_path path/to/system_prompts --output_dir test_results

# Generate just the test plan based on the input file
python test_runner.py --feature_input_file input_file/collective_feature.json --generate_plan-only --output_dir test_results --prompt_path path/to/system_prompts

# just generate tests based on user provided test plan
python test_runner.py --test_plan path/to/plan.json --output_dir path/to/output_dir --prompt_path prompts/collective

# Generate and execute python tests
python test_runner.py --feature_input_file input_file/collective_feature.json --output_dir test_results --prompt_path prompts/collective --execute_python

# Generate and execute python tests based on context_dir
python test_runner.py --feature_input_file input_file/collective_feature.json --output_dir test_results --prompt_path prompts/collective --execute_python --add_context_dir path/to/context_dir

# Build the generated cpp from test plan-> cutalss usecase
python test_runner.py --test_plan template_input_file/cutlass_flash_attention_decode.json --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/flash_attention --build --build_dir cutlass-sycl/build --build_cmd 'ninja cutlass_test_unit_flash_attention_decode_ai' --add_context_dir input_dirs/code/flash_attention/temp

# Just save the generated cpp from test plan using RAG-> cutalss usecase
python test_runner.py --test_plan template_input_file/cutlass_flash_attention_decode.json --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/flash_attention

# Just save the generated cpp from test plan using context_dir -> cutlass usecase
python test_runner.py --test_plan template_input_file/cutlass_flash_attention_decode.json --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/flash_attention --add_context_dir input_dirs/code/flash_attention/temp

#build and execute the generated cpp from test plan-> cutalss usecase
python test_runner.py --test_plan template_input_file/cutlass_flash_attention_decode.json --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/flash_attention --build --build_dir cutlass-sycl/build --build_cmd 'ninja cutlass_test_unit_flash_attention_decode_ai' --add_context_dir input_dirs/code/flash_attention/temp --execute_cpp --execute_dir cutlass-sycl/build/test/unit/flash_attention/flash_attention_decode
```

### working example
#### Python flow -> for collectives.
```bash
# Just generate test cases - don't run them
python test_runner.py --feature_input_file template_input_file/collective_feature.json --prompt_path path/to/system_prompts --output_dir test_results

# Generate and execute python tests
python test_runner.py --feature_input_file template_input_file/collective_feature.json --output_dir test_results --prompt_path prompts/collective --execute_python
```

#### CPP flow -> for cutlass
follow these steps to do this

```bash
#clone sycl-tla
git clone https://github.com/intel/sycl-tla.git
#clone the ValGenAgent repo
git clone https://github.com/pramodkumar-habanalabs/ValGenAgent.git
#go in the ValGenAgent repo
cd ValGenAgent
#make a directory for context lets name it temp_cutlass
mkdir input_dirs/code/context_flashattention_decode
#put the files required in the temp cutlass folder
cp /home/sdp/QA/sycl-tla/test/unit/flash_attention/flash_attention_decode/flash_decode_testbed_3x.hpp input_dirs/code/context_flashattention_decode
cp /home/sdp/QA/sycl-tla/test/unit/flash_attention/flash_attention_decode/xe_flash_decode_bf16_fp32_fp32_h64_512_nonpaged.cpp input_dirs/code/context_flashattention_decode
```
Now set the variables for cutlass building and execution work flow
```bash
#set the variables for cutlass use case for build
source /opt/intel/oneapi/setvars.sh

export CC=icx
export CXX=icpx
export CUTLASS_SYCL_PROFILING_ENABLED=ON
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

# Persist environment variables to following steps **Important step because if its empty or shows error, setup is incomplete or variables not set properly
sycl-ls
```
Now make changes in the cmake for having the build dependencies defined correctly
```bash
#For the case where the ninja build command is 'ninja cutlass_test_unit_flash_attention_decode_ai' we will do the add the following in the cmake

cutlass_test_unit_add_executable(
  cutlass_test_unit_flash_attention_decode_xe_ai
  xe_flash_decode_generated.cpp
)

add_custom_target(
  cutlass_test_unit_flash_attention_decode_ai
  cutlass_test_unit_flash_attention_decode_xe_ai
)
```
 
Now run the agent using the following command
```bash
python test_runner.py --test_plan template_input_file/cutlass_flash_attention_decode.json --output_dir /home/sdp/QA/sycl-tla/test/unit/flash_attention/flash_attention_decode/ --prompt_path prompts/cutlass/flash_attention --build --build_dir /home/sdp/QA/sycl-tla/build --build_cmd 'ninja cutlass_test_unit_flash_attention_decode_ai' --add_context_dir input_dirs/code/context_flashattention_decode --execute_cpp --execute_dir /home/sdp/QA/sycl-tla/build/test/unit/flash_attention/flash_attention_decode/
```

---
## Property Graph RAG and Vector DB RAG

The framework provides flexible options for building and querying retrieval-augmented generation (RAG) databases — either through a **Vector Database (Vector DB)** or a **Property Graph Database (Property Graph RAG)**.  
Both serve different purposes and can be chosen based on the nature of your data, the relationships involved, and the retrieval requirements.

Using the `--index_db` flag, you can select the type of index database to use.  
Additionally, the `--remove_index_db` flag can be applied to clear or rebuild an existing index before creating a new one.

---

### Overview

#### Vector DB RAG
The **Vector Database** option is best suited for unstructured or semi-structured data such as documents, text embeddings, or feature vectors.  
It uses similarity-based search (e.g., cosine similarity, dot product) to retrieve contextually similar entries during RAG operations.

**Use Case Examples:**
- Semantic retrieval for textual or multimodal data.  
- Embedding-based context lookup for prompt augmentation.  
- Fast nearest-neighbor search for large-scale datasets.


#### Property Graph RAG
The **Property Graph RAG** approach is ideal for structured, relational, or interconnected data.  
It models information as nodes and edges with associated properties, allowing for context retrieval that respects graph relationships and dependencies.

**Use Case Examples:**
- Representing entities, relationships, and their attributes.  
- Querying complex dependency graphs for reasoning tasks.  
- Use cases where contextual meaning depends on linked relationships rather than text similarity.

```bash
# with vector_db. Here we add --remove_index_db in order to remove the existing index_db.
python test_runner.py --feature_input_file input_file/collective_feature.json --prompt_path path/to/system_prompts --output_dir test_results index_db vector_db --remove_index_db
# with property_graph
python test_runner.py --feature_input_file input_file/collective_feature.json --prompt_path path/to/system_prompts --output_dir test_results index_db property_graph --remove_index_db
```

---
##  Use cases

This application supports various types of execution plans. Depending on the requirement, you can either:
- Only create the test cases
- Only generate the test plan
- Build the executables if applicable
- Run the complete End-to-End workflow - test plan generation, test code generation, and execution

# Human Feedback Flow

The **Human Feedback Flow** enables a human-in-the-loop workflow where a user can actively guide and manage the agent’s actions during group chat interactions.  
This approach allows dynamic collaboration — users can review, refine, and re-run stages to improve outcomes while preventing the system from automatically repeating the same mistakes.

With this flow, you can generate, review, and regenerate tests in real time, ensuring higher quality and more controlled automation.

---

### Key Use Cases

- Run tests on the initial prompt to evaluate early outputs.  
- Create a test plan first, then execute the generated tests for structured evaluation.  
- Iterate interactively by refining prompts or results based on human feedback.

---
```bash
#No execution just save the file:
python test_runner.py --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/ --human_feedback

#With execution in python
python test_runner.py --feature_input_file input_file/collective_feature.json --output-dir test_results --prompt_path prompts/collective --execute_python

#with test plan generated
python test_runner.py --feature_input_file input_file/collective_feature.json --output-dir test_results --prompt_path prompts/collective --execute_python --with_test_plan

#with execution in cpp
python test_runner.py --output_dir /path/to/folder --prompt_path prompts/cutlass/cutlass_humanfeedback --add_context_dir /path/to/folder --build --build_dir /path/to/folder --build_cmd 'ninja MainloopIntelXeXMX16Group_generated_test' --execute_cpp --execute_dir /path/to/folder

#with verbose enabled. Detailed logs are provided.
python test_runner.py --output_dir cutlass-sycl/test/unit/flash_attention/flash_attention_decode --prompt_path prompts/cutlass/ --human_feedback --verbose
```

## Web based

We provide a user interface to use this tool. It offers simple interactive UI that has currently support what we user can do using command line.

### Steps to start the application:
1. First clone the repo on your personal vm
   ```bash
   git clone https://github.com/mansi05ag/ValGenAgent.git
   cd ValGenAgent
   pip install -r requirements.txt
   ```
2. Now go inside the directory webapp
   ```bash
   cd webapp
   ```
3. After that run the application

   ```bash
   python app.py --common_dir <path to common directory>
   ```
   > Note: For running the application we must provide a common directory accessible by both the vm and the device you are connecting to.
5. Access the application using port no. 8002 through url.


#  How to Use the web application

Steps to run the application using UI:

---

##  Step 1: Connect to the Target Container or Device

- Click the **Connect** button on your virtual machine (VM) interface to initiate the container setup.
- This will establish a connection to the required container or device.

- If you're working with an **`hlctl` container** in the **`qa` namespace**, you can also connect manually using the following command:

  ```bash
  hlctl container exec -w <container_name> bash -n qa
  ```

  >  Replace `<container_name>` with your actual container name.

> **Note:** The *Container Name* input field is relevant **only if you're using an `hlctl` container with the `qa` namespace**.

---

##  Step 2: Upload Files and Select Functionality

- Upload all required files.
  - Your **code and documentation** should be combined into a **single `.zip` file**.
- Choose the functionality you want the application to perform from the available options.
- Click the **Run** button to start execution.

>  The application will now run on the connected container or device, and you’ll see real-time output and logs on the screen.

---

##  Step 3: Download the Output

- Once execution completes successfully, a **ZIP file** containing the **generated output or results** will be available.
- Click on the **Download** button to save the output to your system.

---
