.. _release-verification:

Verifying release integrity
============================

This document explains how to verify that a Grid2Op release asset (``.whl`` or
``.tar.gz``) is authentic and has not been tampered with.

Two methods are available depending on the version you are verifying:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Grid2Op version
     - Recommended method
   * - ≥ 1.12.5
     - :ref:`Sigstore attestations via PyPI <sigstore-method>`
   * - < 1.12.5
     - :ref:`Digest check via check-release <digest-method>`

Both methods are independent of each other. For versions ≥ 1.12.5 you may
use either or both.

----

.. _sigstore-method:

Method 1 — Sigstore attestations (Grid2Op ≥ 1.12.5)
-----------------------------------------------------

Starting from version 1.12.5, Grid2Op releases are signed using
`Sigstore <https://www.sigstore.dev/>`_ via PyPI's attestation mechanism.
This cryptographically proves that a release asset was built by the official
Grid2Op GitHub Actions pipeline and has not been modified since.

.. warning::

   This method does not work for versions prior to 1.12.5.
   Use :ref:`Method 2 <digest-method>` for those.

Prerequisites
~~~~~~~~~~~~~

.. code-block:: bash

   pip install pypi-attestations

Verify a wheel or sdist
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pypi_attestations verify pypi \
     --repository https://github.com/Grid2op/grid2op \
     Grid2Op-<VERSION>-py3-none-any.whl

Replace ``<VERSION>`` with the release you are verifying, for example ``1.12.5``.
The same command works for ``.tar.gz`` files.

Expected output
~~~~~~~~~~~~~~~

.. code-block:: text

   OK: Grid2Op-1.12.5-py3-none-any.whl

If verification fails, the tool will print an error. Do not use the file.

What is being verified
~~~~~~~~~~~~~~~~~~~~~~

- The asset was produced by the ``release.yml`` workflow in the
  ``Grid2op/grid2op`` GitHub repository.
- The asset has not been modified since it was uploaded to PyPI.
- The signing identity matches the Grid2Op repository — not just any
  GitHub Actions run.

You can also inspect the attestation directly on the PyPI release page at
``https://pypi.org/project/Grid2Op/<VERSION>/`` under the **Provenance**
section.

Expected Sigstore identity
~~~~~~~~~~~~~~~~~~~~~~~~~~

When verifying a Grid2Op release ≥ 1.12.5, the Sigstore certificate must
match **all** of the following values exactly. Any deviation should be treated
as a verification failure.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Expected value
   * - **Issuer**
     - ``https://token.actions.githubusercontent.com``
   * - **Subject (workflow identity)**
     - ``https://github.com/Grid2op/grid2op/.github/workflows/release.yml@refs/tags/v*``
   * - **Source repository**
     - ``https://github.com/Grid2op/grid2op``

The ``*`` in the subject matches any version tag (e.g. ``v1.12.5``,
``v2.0.0``). The issuer confirms the signature came from GitHub Actions
specifically, not any other Sigstore-compatible system.

If the ``pypi_attestations verify`` command succeeds but the certificate
fields above do not match when inspected manually, do not trust the asset.
To inspect the certificate fields directly:

.. code-block:: bash

   python -m pypi_attestations inspect \
     Grid2Op-<VERSION>-py3-none-any.whl

----

.. _digest-method:

Method 2 — Digest check (legacy releases < 1.12.5)
----------------------------------------------------

For releases prior to 1.12.5, integrity can be verified by comparing the
asset's cryptographic digest against the value recorded by PyPI at upload
time.

Grid2Op ships a built-in command for this:

.. code-block:: bash

   python -m grid2op check-release <path-to-asset>

Prerequisites
~~~~~~~~~~~~~

A working Grid2Op installation (any version). No extra dependencies are
required — the command uses only the Python standard library.

Basic usage
~~~~~~~~~~~

.. code-block:: bash

   # Default: verifies BLAKE2b-256 digest against live PyPI
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl

Choosing which digest algorithm to check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # BLAKE2b-256 (default, strongest)
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl --blake2b

   # SHA-256
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl --sha256

   # MD5 (weak — use only if the above are unavailable)
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl --md5

   # All three at once
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl --all

Expected output
~~~~~~~~~~~~~~~

.. code-block:: text

   File    : Grid2Op-1.9.8-py3-none-any.whl
   Version : 1.9.8
   Source  : PyPI (live)
   Checking: blake2b_256

     BLAKE2B-256      PASS

   ✓ All checks passed.

If any digest does not match, the tool prints the expected and actual values
and exits with a non-zero status. Do not use the file.

Offline verification (air-gapped environments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you cannot reach PyPI, first generate a local digest database on a
connected machine using the helper script in the repository:

.. code-block:: bash

   # On a machine with internet access — run once, or per release
   python scripts/fetch_release_digests.py --output digests.json

   # Transfer digests.json to the air-gapped machine, then:
   python -m grid2op check-release Grid2Op-1.9.8-py3-none-any.whl \
     --offline digests.json

The ``fetch_release_digests.py`` script records all historical releases by
default, so a single ``digests.json`` file covers all versions.

What is being verified
~~~~~~~~~~~~~~~~~~~~~~

The digest recorded by PyPI at the time of upload is compared against a
freshly computed digest of the local file. A mismatch means the file has
been modified after upload. Because the digests are fetched directly from
PyPI (or from a snapshot you generated yourself), this method does not rely
on any file bundled inside the Grid2Op package.

Expected identity for legacy releases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For releases prior to 1.12.5, there is no cryptographic workflow identity.
The expected identity is the PyPI account that owns the ``Grid2Op`` package:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Expected value
   * - **PyPI package name**
     - ``Grid2Op`` (https://pypi.org/project/Grid2Op/)
   * - **PyPI owning organisation**
     - ``Grid2op`` (https://pypi.org/org/Grid2op/)
   * - **PyPI collaborator**
     - ``lfeoperations`` — LF Energy Operations (https://pypi.org/user/lfeoperations/)
   * - **Source repository**
     - ``https://github.com/Grid2op/grid2op``

The digests returned by ``check-release`` are fetched directly from the
PyPI JSON API for the above package. Verifying that the package name and
owning organisation match before trusting the digests is recommended.

----

Summary
-------

.. code-block:: text

   Is your version ≥ 1.12.5?

     YES → Method 1 (Sigstore)
           pip install pypi-attestations
           python -m pypi_attestations verify pypi \
             --repository https://github.com/Grid2op/grid2op \
             Grid2Op-<VERSION>-py3-none-any.whl

     NO  → Method 2 (digest)
           python -m grid2op check-release \
             Grid2Op-<VERSION>-py3-none-any.whl

If you have concerns about a release, please open an issue at
https://github.com/Grid2op/grid2op/issues or follow the process described
in :ref:`security` for security-sensitive matters.
