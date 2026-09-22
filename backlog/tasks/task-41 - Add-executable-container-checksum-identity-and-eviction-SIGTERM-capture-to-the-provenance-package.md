---
id: TASK-41
title: >-
  Add executable/container checksum identity and eviction (SIGTERM) capture to
  the provenance package
status: To Do
assignee: []
created_date: '2026-09-22 15:13'
labels: []
dependencies:
  - TASK-31
  - TASK-38
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Beyond zero-config site/jobad capture (task-38), the provenance package can also verify the identity of what actually ran, and capture eviction as a distinct event -- both require more research/empirical verification against live HTCondor internals than task-38's scope, so they're tracked separately and build on task-38's auto-run hook.

Software/executable identity: an executable checksum computed both at submission time and at execution time (catching multi-site transfer corruption as a side effect of identity-checking); and, for containers, an actual checksum rather than a mutable name/tag/path wherever one is achievable. Names and tags are not solid identity -- a checksum is. HTCondor's container universe docs confirm a locally-supplied .sif image is transferred by default into the job's own scratch directory, which is then bind-mounted into the running container at the same path -- so the exact .sif bytes used to mount the container are typically hashable from inside the container itself, no runtime cooperation required. That does not hold for docker://-/oras://-referenced images (pulled directly by the runtime into its own cache, never transferred by HTCondor) or for genuine Docker universe: HTCondor's own docker_proc.cpp source confirms HTCondor never resolves or records an image digest anywhere, and nothing inside a running Docker container reveals its own image digest without daemon-socket access. For genuine Docker universe the only honest path to a real digest is AP-side, at submission time, via a plain registry API call resolving the submitted tag -- disclosed as "what the tag pointed to at submission time," not a runtime-verified guarantee.

Lifecycle: a SIGTERM handler (HTCondor's default vacate/eviction signal) in addition to atexit, so eviction mid-run -- the entire reason this multi-site provenance system exists -- is captured as a distinct event instead of silently lost.

Empirical findings from a real container-universe job (Docker backend, via container_image = docker://hub.opensciencegrid.org/xdd/metl_gb1:2026-06-23-2), carried forward from this work's original research (formerly task-38):
- /.dockerenv confirms condor_ssh_to_job did land inside the container.
- /proc/1/cgroup showing '0::/' is NOT a useful runtime signal -- under cgroup v2 with a cgroup namespace, a contained process always sees its own namespace-relative root regardless of runtime. Drop that check.
- The job ad carries ContainerImage ("docker://hub.opensciencegrid.org/xdd/metl_gb1:2026-06-23-2") AND ContainerImageSource ("docker") -- a cleaner dispatch signal than string-sniffing container_image's prefix/suffix the way mldag/annex/create.py's extract_sif_file() currently does (which only checks for a .sif suffix and would silently return None for this exact job).
- This job uses a mutable tag (2026-06-23-2), not a pinned digest -- confirms the no-digest-available gap for genuine/Docker-backed container jobs still applies here.

Open questions to confirm with HTCondor developers before finalizing AC#2-4 (in flight):
1. Does container_image = docker://...@sha256:<digest> (digest-pinned) work cleanly through the container-universe abstraction the same way digest-pinning works for legacy docker_image?
2. What other values can ContainerImageSource take (presumably apptainer/singularity for the .sif case) -- is it documented/stable enough to rely on as a dispatch key?
3. Is there ever a resolved-digest ClassAd attribute written back after the image is pulled, that wasn't found in the docs/source reviewed so far?
Do not finalize AC#2-4 until this comes back -- a confirmed ContainerImageSource switch plus a confirmed digest-pinning path would meaningfully simplify both versus the current string-sniffing/registry-resolution approach.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Computes and records a checksum of the training executable both at submission time (pre.py, hashing the AP-local file before transfer, reusing sidecar.py's existing sha256_file()) and at execution time (hashing the transferred copy inside the job sandbox), so a mismatch surfaces transfer corruption
- [ ] #2 For Apptainer/Singularity containers using a locally-supplied .sif image (container_image pointing at a file, transferred into the job's own scratch directory under HTCondor's default transfer_container behavior), computes a checksum of the actual .sif file from inside the running container, dispatching on the ContainerImageSource ClassAd attribute -- confirmed more reliable than extract_sif_file()'s current .sif-suffix string-sniffing
- [ ] #3 Empirically verifies, on a real execute node, whether a docker://- or oras://-referenced image (pulled directly by the runtime into its own local cache rather than transferred by HTCondor as a file) is reachable/hashable from inside the running container; if not reachable, records the best available identity (SINGULARITY_CONTAINER/APPTAINER_CONTAINER plus /.singularity.d/labels.json) with an explicit caveat that it is not a verified checksum of what ran
- [ ] #4 For genuine Docker-universe jobs, resolves the submitted docker_image tag to a registry manifest digest via a plain HTTPS registry API call at submission time (in pre.py, no daemon/socket needed), recorded as the digest the tag pointed to at submission -- not verified against what the execute node actually pulled
- [ ] #5 Registers a SIGTERM handler (HTCondor's default vacate/eviction signal) in addition to atexit, so eviction/preemption mid-run is captured as a distinct real-time event, built on task-38's auto-run hook
- [ ] #6 All additions are best-effort: any failure (unreadable executable, unreachable registry, non-HTCondor environment) is caught and never alters the training job's behavior, output, or exit code
- [ ] #7 backlog/docs/provenance_design.md documents this tier, including exactly which container cases get a verified checksum vs. a disclosed best-effort identity
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm with HTCondor developers the open questions above (digest-pinned docker://...@sha256:<digest> support through container-universe; stable ContainerImageSource value set; any resolved-digest ClassAd attribute) before finalizing AC#2-4
2. Implement submission-time and execution-time executable checksums and the comparison, reusing sidecar.py's sha256_file()
3. Verify (via the job's own process, not an interactive ssh session) whether a locally-supplied .sif's bind-mounted scratch-dir copy is hashable from inside the container as expected -- reuse the TransferContainer/ContainerImage/iwd resolution logic already in mldag/annex/create.py's extract_sif_file() -- and separately verify whether a docker://-/oras://-pulled image's runtime cache is reachable from inside the container; implement whichever checksum path each case actually supports, with the SINGULARITY_CONTAINER/APPTAINER_CONTAINER + labels.json fallback for cases where it isn't
4. Implement the Docker-universe submission-time registry-digest resolution in pre.py (plain HTTPS registry API call, no daemon/socket)
5. Implement SIGTERM handling alongside atexit, building on task-38's auto-run hook
6. Update backlog/docs/provenance_design.md to document this tier, including exactly which container cases get a verified checksum vs. a disclosed best-effort identity
7. End-to-end test on a real execute node, including a container job (Apptainer, both local-.sif and docker://-pulled) and a simulated eviction (SIGTERM) mid-job
<!-- SECTION:PLAN:END -->
