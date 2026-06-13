import asyncio
import json
from pathlib import Path

from src.dispatcher import make_pipeline
from src.bootstrap import setup_runtime
from src.session import SessionState


async def main():
    default_contract = {
        "id": "default_contract",
        "file_name": "default_contract.txt",
        "text": "This is the DEFAULT contract. If you see this in the result, override failed.",
        "spans": [[0, 78]],
    }

    dynamic_contract_text = """
    NON-DISCLOSURE AGREEMENT

    This Agreement is made between Alpha Company and Beta Company.

    The Receiving Party shall keep all Confidential Information strictly confidential.
    Confidential Information may be disclosed orally, visually, electronically, or in writing.
    The Receiving Party may disclose Confidential Information only to employees and representatives
    who have a need to know and are bound by confidentiality obligations.

    The Receiving Party shall not disclose Confidential Information to any third party without consent.
    If legally compelled to disclose Confidential Information, the Receiving Party shall promptly notify
    the Disclosing Party where legally permitted.

    Upon request, the Receiving Party shall return or destroy Confidential Information.
    The obligations of confidentiality shall survive termination for three years.
    """

    contract_override = {
        "id": "dynamic_user_contract_001",
        "file_name": "dynamic_user_contract.txt",
        "text": dynamic_contract_text,
        "spans": [[0, len(dynamic_contract_text)]],
    }

    session = SessionState.create(
        contract_id="dynamic_user_contract_001",
        contract=contract_override,
        dev_mode=True,
    )

    pipeline = make_pipeline(
        contract=default_contract,
        session_id=session.session_id,
    )

    setup_runtime(
        session=session,
        pipeline=pipeline,
        approval_gate=None,
        use_real_rag=True,
    )

    result = await pipeline(
        contract_id="dynamic_user_contract_001",
        retrieval_mode="vector",
        contract_override=contract_override,
    )

    print("DONE")
    print("Contract ID:", result.get("contract_id"))
    print("Number of hypothesis traces:", len(result.get("hypothesis_traces", [])))
    print("Metrics:", result.get("metrics"))

    Path("dynamic_test_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())