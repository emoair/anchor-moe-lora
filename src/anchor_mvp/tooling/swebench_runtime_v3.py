"""Formal SWE-bench v3 runtime adapter.

This is the only runtime interface that the formal coordinator may use.  It
materialises ``/testbed`` from the official TestSpec instance image, runs the
model against that same canonical worktree, then evaluates the final cumulative
binary diff in a fresh network-none instance container.  Official test material,
test output, grading reports, and authenticated receipts remain under a
supervisor-owned private root and are never returned in model context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol

from .models import AgentExecution
from .policy import ToolPolicy
from .trace import (
    classify_error_metadata,
    digest_text,
    parse_opencode_jsonl,
    parse_public_outcome,
)
from .swebench_execution_v3 import (
    DISTILLATION_EXECUTION_BINDING_KEYS,
    EXECUTION_TOOL_CONTRACT_V3,
    ExecutionContractError,
    distillation_tool_evidence,
    distillation_validation_state_sha256,
    load_execution_lock,
    official_harness_import_scope,
    resolve_official_instance_image_key,
    sign_distillation_execution_receipt,
    sign_official_eval_receipt,
    verify_official_eval_receipt,
)


RUNTIME_ADAPTER_VERSION = "anchor.swebench-runtime-adapter.v3"
IMAGE_ACQUISITION_REQUEST_SCHEMA = "anchor.swebench-image-acquisition-request.v1"
IMAGE_CACHE_BINDING_SCHEMA = "anchor.swebench-image-cache-binding.v1"
IMAGE_CACHE_LEDGER_SCHEMA = "anchor.swebench-image-cache-ledger.v1"
CANONICAL_TESTBED = "/testbed"
SUPERVISOR_RECEIPT_KEY_PATH = "/var/lib/anchor/keys/official-eval-hmac-v1"
DISTILLATION_RECEIPT_KEY_PATH = (
    "/var/lib/anchor/keys/distillation-execution-hmac-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_IMAGE_KEY = re.compile(r"^[a-z0-9][a-z0-9._/:\-]{1,511}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NATIVE_ROOT = re.compile(r"^/(?:var/lib|home)/[A-Za-z0-9_./-]+$")
_SESSION_ID = re.compile(r"^ses_[A-Za-z0-9_-]{4,128}$")
_PLATFORM = re.compile(r"^linux/(?:x86_64|amd64|arm64(?:/v8)?)$")


def _exact_mapping(value: Mapping[str, Any], fields: set[str], code: str) -> None:
    if set(value) != fields:
        raise ExecutionContractError(code)


def _image_recipe_value(spec: object, name: str) -> str:
    value = getattr(spec, name, None)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ExecutionContractError("v3_runtime_image_recipe_invalid")
    return value


@dataclass(frozen=True)
class OfficialImageAcquisitionRequest:
    """Private supervisor request binding one task to one official image recipe."""

    execution_lock_sha256: str
    dataset_revision: str
    task_id_sha256: str
    instance_id_sha256: str
    base_commit: str
    image_key: str
    base_image_key: str
    env_image_key: str
    platform: str
    base_dockerfile: str
    env_dockerfile: str
    instance_dockerfile: str
    setup_env_script: str
    setup_repo_script: str

    @classmethod
    def from_test_spec(
        cls,
        *,
        execution_lock_sha256: str,
        dataset_revision: str,
        task_id: str,
        instance_id: str,
        base_commit: str,
        image_key: str,
        test_spec: object,
    ) -> "OfficialImageAcquisitionRequest":
        base_image_key = getattr(test_spec, "base_image_key", None)
        env_image_key = getattr(test_spec, "env_image_key", None)
        platform = getattr(test_spec, "platform", None)
        if (
            not _SHA256.fullmatch(execution_lock_sha256)
            or not _SHA256.fullmatch(task_id.rsplit(":", 1)[-1])
            or not _INSTANCE_ID.fullmatch(instance_id)
            or not _COMMIT.fullmatch(base_commit)
            or not isinstance(dataset_revision, str)
            or not _COMMIT.fullmatch(dataset_revision)
            or not isinstance(base_image_key, str)
            or not _IMAGE_KEY.fullmatch(base_image_key)
            or not isinstance(env_image_key, str)
            or not _IMAGE_KEY.fullmatch(env_image_key)
            or not _IMAGE_KEY.fullmatch(image_key)
            or not isinstance(platform, str)
            or not _PLATFORM.fullmatch(platform)
        ):
            raise ExecutionContractError("v3_runtime_image_acquisition_binding_invalid")
        return cls(
            execution_lock_sha256=execution_lock_sha256,
            dataset_revision=dataset_revision,
            task_id_sha256=_sha256(task_id.encode("utf-8")),
            instance_id_sha256=_sha256(instance_id.encode("utf-8")),
            base_commit=base_commit,
            image_key=image_key,
            base_image_key=base_image_key,
            env_image_key=env_image_key,
            platform=platform,
            base_dockerfile=_image_recipe_value(test_spec, "base_dockerfile"),
            env_dockerfile=_image_recipe_value(test_spec, "env_dockerfile"),
            instance_dockerfile=_image_recipe_value(test_spec, "instance_dockerfile"),
            setup_env_script=_image_recipe_value(test_spec, "setup_env_script"),
            setup_repo_script=_image_recipe_value(test_spec, "install_repo_script"),
        )

    def recipe_sha256(self) -> str:
        return _sha256(
            _canonical(
                {
                    "base_image_key": self.base_image_key,
                    "env_image_key": self.env_image_key,
                    "platform": self.platform,
                    "base_dockerfile": self.base_dockerfile,
                    "env_dockerfile": self.env_dockerfile,
                    "instance_dockerfile": self.instance_dockerfile,
                    "setup_env_script": self.setup_env_script,
                    "setup_repo_script": self.setup_repo_script,
                }
            ).encode("utf-8")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": IMAGE_ACQUISITION_REQUEST_SCHEMA,
            "execution_lock_sha256": self.execution_lock_sha256,
            "dataset_revision": self.dataset_revision,
            "task_id_sha256": self.task_id_sha256,
            "instance_id_sha256": self.instance_id_sha256,
            "base_commit": self.base_commit,
            "image_key": self.image_key,
            "base_image_key": self.base_image_key,
            "env_image_key": self.env_image_key,
            "platform": self.platform,
            "base_dockerfile": self.base_dockerfile,
            "env_dockerfile": self.env_dockerfile,
            "instance_dockerfile": self.instance_dockerfile,
            "setup_env_script": self.setup_env_script,
            "setup_repo_script": self.setup_repo_script,
            "recipe_sha256": self.recipe_sha256(),
        }


@dataclass(frozen=True)
class OfficialImageCacheBinding:
    """Authenticated-by-recomputation cache result returned by the supervisor."""

    execution_lock_sha256: str
    dataset_revision: str
    task_id_sha256: str
    instance_id_sha256: str
    base_commit: str
    image_key: str
    image_digest: str
    image_ref: str
    recipe_sha256: str
    acquisition_mode: str
    binding_sha256: str
    ledger_content_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfficialImageCacheBinding":
        fields = {
            "schema_version",
            "execution_lock_sha256",
            "dataset_revision",
            "task_id_sha256",
            "instance_id_sha256",
            "base_commit",
            "image_key",
            "image_digest",
            "image_ref",
            "recipe_sha256",
            "acquisition_mode",
            "binding_sha256",
            "ledger_content_sha256",
        }
        _exact_mapping(value, fields, "v3_runtime_image_cache_binding_shape_invalid")
        unsigned = {name: value[name] for name in fields - {"schema_version", "binding_sha256", "ledger_content_sha256"}}
        expected = _sha256(_canonical(unsigned).encode("utf-8"))
        if (
            value.get("schema_version") != IMAGE_CACHE_BINDING_SCHEMA
            or any(
                not isinstance(value.get(name), str)
                or not _SHA256.fullmatch(str(value.get(name)))
                for name in (
                    "execution_lock_sha256",
                    "task_id_sha256",
                    "instance_id_sha256",
                    "recipe_sha256",
                    "binding_sha256",
                    "ledger_content_sha256",
                )
            )
            or not isinstance(value.get("dataset_revision"), str)
            or not _COMMIT.fullmatch(str(value.get("dataset_revision")))
            or not isinstance(value.get("base_commit"), str)
            or not _COMMIT.fullmatch(str(value.get("base_commit")))
            or not isinstance(value.get("image_key"), str)
            or not _IMAGE_KEY.fullmatch(str(value.get("image_key")))
            or not isinstance(value.get("image_digest"), str)
            or not _IMAGE_DIGEST.fullmatch(str(value.get("image_digest")))
            or not isinstance(value.get("image_ref"), str)
            or not (
                str(value.get("image_ref")) == str(value.get("image_digest"))
                or str(value.get("image_ref")).endswith("@" + str(value.get("image_digest")))
            )
            or value.get("acquisition_mode") not in {"pull", "official-recipe-build"}
            or value.get("binding_sha256") != expected
        ):
            raise ExecutionContractError("v3_runtime_image_cache_binding_invalid")
        return cls(**{name: str(value[name]) for name in fields - {"schema_version"}})

    def matches(self, request: OfficialImageAcquisitionRequest) -> bool:
        return bool(
            self.execution_lock_sha256 == request.execution_lock_sha256
            and self.dataset_revision == request.dataset_revision
            and self.task_id_sha256 == request.task_id_sha256
            and self.instance_id_sha256 == request.instance_id_sha256
            and self.base_commit == request.base_commit
            and self.image_key == request.image_key
            and self.recipe_sha256 == request.recipe_sha256()
        )

_HOST_UNIX_ROUTE_RELAY = r"""
import os,socket,sys,threading
path,host,port=sys.argv[1],sys.argv[2],int(sys.argv[3])
try: os.unlink(path)
except FileNotFoundError: pass
server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
server.bind(path)
try: os.chmod(path,0o600)
except OSError:
    if os.name!='nt': raise
server.listen(32)
def handle(client):
    upstream=None
    try:
        head=b''
        while b'\r\n\r\n' not in head and len(head)<65536:
            chunk=client.recv(4096)
            if not chunk: return
            head+=chunk
        header,sep,rest=head.partition(b'\r\n\r\n')
        if not sep: return
        lines=header.split(b'\r\n')
        parts=lines[0].split(b' ')
        if len(parts)!=3 or parts[0].upper()==b'CONNECT': return
        target=parts[1]
        route=target.split(b'?',1)[0]
        if route not in {b'/anchor/health',b'/anchor/v1/responses',b'/v1/models',b'/v1/responses'}: return
        host_headers=[line.split(b':',1)[1].strip().lower() for line in lines[1:] if line.lower().startswith(b'host:')]
        allowed={b'127.0.0.1',b'127.0.0.1:18080',b'localhost',b'localhost:18080'}
        if len(host_headers)!=1 or host_headers[0] not in allowed: return
        if any(line.lower().startswith((b'transfer-encoding:',b'expect:')) for line in lines[1:]): return
        lengths=[line.split(b':',1)[1].strip() for line in lines[1:] if line.lower().startswith(b'content-length:')]
        if len(lengths)>1: return
        try: content_length=int(lengths[0]) if lengths else 0
        except ValueError: return
        if content_length<0 or content_length>33554432 or len(rest)>content_length: return
        body=rest
        while len(body)<content_length:
            chunk=client.recv(min(65536,content_length-len(body)))
            if not chunk: return
            body+=chunk
        forwarded=[line for line in lines[1:] if not line.lower().startswith((b'connection:',b'proxy-connection:'))]
        forwarded.append(b'Connection: close')
        request=b' '.join(parts)+b'\r\n'+b'\r\n'.join(forwarded)+b'\r\n\r\n'+body
        upstream=socket.create_connection((host,port),timeout=10)
        upstream.sendall(request)
        upstream.shutdown(socket.SHUT_WR)
        while True:
            chunk=upstream.recv(65536)
            if not chunk: break
            client.sendall(chunk)
    finally:
        try: client.close()
        except OSError: pass
        if upstream is not None:
            try: upstream.close()
            except OSError: pass
while True:
    client,_=server.accept()
    threading.Thread(target=handle,args=(client,),daemon=True).start()
"""

_CONTAINER_TCP_UNIX_BRIDGE = r"""
import socket,sys,threading
path,port=sys.argv[1],int(sys.argv[2])
def pump(src,dst):
    try:
        while True:
            chunk=src.recv(65536)
            if not chunk: break
            dst.sendall(chunk)
    except OSError: pass
    try: dst.shutdown(socket.SHUT_WR)
    except OSError: pass
def handle(client):
    upstream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        upstream.connect(path)
        a=threading.Thread(target=pump,args=(client,upstream),daemon=True)
        b=threading.Thread(target=pump,args=(upstream,client),daemon=True)
        a.start(); b.start(); a.join(); b.join()
    finally:
        client.close(); upstream.close()
server=socket.socket(); server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.bind(('127.0.0.1',port)); server.listen(32)
while True:
    client,_=server.accept()
    threading.Thread(target=handle,args=(client,),daemon=True).start()
"""


# Executed only as root inside the locked WSL supervisor.  It deliberately
# contains the only `podman pull` / networked build path in the formal runtime.
# Model and evaluator paths below continue to use --network=none and
# --pull=never with the returned immutable reference.
_SUPERVISOR_IMAGE_CACHE = r"""
import fcntl,hashlib,json,os,pathlib,re,shutil,stat,subprocess,sys,tempfile
ROOT=pathlib.Path(sys.argv[1])
REQUEST=pathlib.Path(sys.argv[2])
SCHEMA='anchor.swebench-image-cache-ledger.v1'
BINDING_SCHEMA='anchor.swebench-image-cache-binding.v1'
REQUEST_SCHEMA='anchor.swebench-image-acquisition-request.v1'
SHA=re.compile(r'^[0-9a-f]{64}$')
DIGEST=re.compile(r'^sha256:[0-9a-f]{64}$')
COMMIT=re.compile(r'^[0-9a-f]{40}$')
IMAGE=re.compile(r'^[a-z0-9][a-z0-9._/:\-]{1,511}$')
PLATFORM=re.compile(r'^linux/(?:x86_64|amd64|arm64(?:/v8)?)$')
REQUEST_FIELDS={
 'schema_version','execution_lock_sha256','dataset_revision','task_id_sha256',
 'instance_id_sha256','base_commit','image_key','base_image_key','env_image_key',
 'platform','base_dockerfile','env_dockerfile','instance_dockerfile',
 'setup_env_script','setup_repo_script','recipe_sha256'
}
BINDING_FIELDS={
 'execution_lock_sha256','dataset_revision','task_id_sha256','instance_id_sha256',
 'base_commit','image_key','image_digest','image_ref','recipe_sha256',
 'acquisition_mode','binding_sha256'
}
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(v): return hashlib.sha256(v).hexdigest()
def die(code):
 print(canonical({'schema_version':BINDING_SCHEMA,'error_code':code}))
 raise SystemExit(3)
def secure_dir(path,create=False):
 if create:
  try: path.mkdir(parents=True,exist_ok=True,mode=0o700)
  except Exception: die('image_cache_private_root_unavailable')
 try: value=path.lstat()
 except Exception: die('image_cache_private_root_unavailable')
 if (not stat.S_ISDIR(value.st_mode) or value.st_uid!=0 or
  stat.S_IMODE(value.st_mode)!=0o700): die('image_cache_private_root_insecure')
def secure_regular(path):
 try: value=path.lstat()
 except Exception: die('image_cache_ledger_invalid')
 if (not stat.S_ISREG(value.st_mode) or value.st_uid!=0 or
  stat.S_IMODE(value.st_mode)!=0o600): die('image_cache_ledger_permissions_invalid')
def open_private_lock(path):
 flags=os.O_RDWR|os.O_CREAT
 if hasattr(os,'O_NOFOLLOW'): flags|=os.O_NOFOLLOW
 try:
  fd=os.open(path,flags,0o600); value=os.fstat(fd)
 except Exception: die('image_cache_lock_unavailable')
 if (not stat.S_ISREG(value.st_mode) or value.st_uid!=0 or
  stat.S_IMODE(value.st_mode)!=0o600):
  os.close(fd); die('image_cache_lock_permissions_invalid')
 return os.fdopen(fd,'a+b')
def run(args,timeout=7200):
 return subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
  stderr=subprocess.DEVNULL,timeout=timeout,check=False)
def image_state(reference):
 p=run(['podman','image','inspect','--format','{{.Digest}}|{{.Id}}',reference],60)
 if p.returncode: return None
 parts=p.stdout.decode('utf-8','replace').strip().split('|',1)
 if len(parts)!=2: return None
 manifest,image_id=parts
 if DIGEST.fullmatch(manifest): return manifest
 if DIGEST.fullmatch(image_id): return image_id
 return None
def binding_hash(entry):
 return sha(canonical({k:entry[k] for k in BINDING_FIELDS-{'binding_sha256'}}).encode())
def ledger_content(value):
 unsigned={k:value[k] for k in ('schema_version','execution_lock_sha256',
  'dataset_revision','entry_count','entries')}
 return sha(canonical(unsigned).encode())
def load_ledger(path,request):
 if not path.exists():
  value={'schema_version':SCHEMA,
   'execution_lock_sha256':request['execution_lock_sha256'],
   'dataset_revision':request['dataset_revision'],'entry_count':0,'entries':{}}
  value['content_sha256']=ledger_content(value)
  return value
 secure_regular(path)
 try: value=json.loads(path.read_text(encoding='utf-8'))
 except Exception: die('image_cache_ledger_invalid')
 if set(value)!={'schema_version','execution_lock_sha256','dataset_revision',
  'entry_count','entries','content_sha256'}: die('image_cache_ledger_shape_invalid')
 entries=value.get('entries')
 if (value.get('schema_version')!=SCHEMA or
  value.get('execution_lock_sha256')!=request['execution_lock_sha256'] or
  value.get('dataset_revision')!=request['dataset_revision'] or
  not isinstance(entries,dict) or value.get('entry_count')!=len(entries) or
  value.get('content_sha256')!=ledger_content(value)):
  die('image_cache_ledger_binding_invalid')
 for key,entry in entries.items():
  if (not SHA.fullmatch(str(key)) or not isinstance(entry,dict) or
   set(entry)!=BINDING_FIELDS or entry.get('task_id_sha256')!=key or
   entry.get('binding_sha256')!=binding_hash(entry)):
   die('image_cache_ledger_entry_invalid')
 return value
def verify_entry(entry,request):
 expected={k:request[k] for k in ('execution_lock_sha256','dataset_revision',
  'task_id_sha256','instance_id_sha256','base_commit','image_key','recipe_sha256')}
 if any(entry.get(k)!=v for k,v in expected.items()): die('image_cache_resume_binding_mismatch')
 digest=entry.get('image_digest'); ref=entry.get('image_ref')
 if not DIGEST.fullmatch(str(digest)): die('image_cache_digest_invalid')
 if ref!=digest and not str(ref).endswith('@'+str(digest)): die('image_cache_ref_invalid')
 if image_state(str(ref))!=digest: die('image_cache_immutable_ref_missing')
 if image_state(request['image_key'])!=digest: die('image_cache_label_drift')
 return entry
def write_atomic(path,value):
 secure_dir(path.parent,create=True)
 temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 data=(canonical(value)+'\n').encode()
 fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 try:
  os.write(fd,data); os.fsync(fd)
 finally: os.close(fd)
 os.replace(temp,path); os.chmod(path,0o600)
 secure_regular(path)
 dfd=os.open(path.parent,os.O_RDONLY)
 try: os.fsync(dfd)
 finally: os.close(dfd)
def context(directory,dockerfile,scripts):
 directory.mkdir(parents=True,exist_ok=False,mode=0o700)
 (directory/'Dockerfile').write_text(dockerfile,encoding='utf-8')
 for name,value in scripts.items(): (directory/name).write_text(value,encoding='utf-8')
def build(request):
 build_lock=ROOT/'image-cache'/'build.lock'
 secure_dir(build_lock.parent,create=True)
 with open_private_lock(build_lock) as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  scratch=pathlib.Path(tempfile.mkdtemp(prefix='official-build-',dir=str(build_lock.parent)))
  try:
   stages=(
    ('base',request['base_image_key'],request['base_dockerfile'],{},'always'),
    ('env',request['env_image_key'],request['env_dockerfile'],
     {'setup_env.sh':request['setup_env_script']},'never'),
    ('instance',request['image_key'],request['instance_dockerfile'],
     {'setup_repo.sh':request['setup_repo_script']},'never'))
   for name,key,dockerfile,scripts,pull in stages:
    target=scratch/name; context(target,dockerfile,scripts)
    command=['podman','build','--network=host','--no-cache','--pull='+pull,
     '--platform',request['platform'],'--tag',key,str(target)]
    if run(command).returncode: die('official_image_recipe_build_failed')
  finally: shutil.rmtree(scratch,ignore_errors=True)
 return image_state(request['image_key'])
try: request=json.loads(REQUEST.read_text(encoding='utf-8'))
except Exception: die('image_acquisition_request_invalid')
if set(request)!=REQUEST_FIELDS or request.get('schema_version')!=REQUEST_SCHEMA:
 die('image_acquisition_request_shape_invalid')
recipe={k:request[k] for k in ('base_image_key','env_image_key','platform',
 'base_dockerfile','env_dockerfile','instance_dockerfile','setup_env_script',
 'setup_repo_script')}
if (not all(SHA.fullmatch(str(request.get(k,''))) for k in
 ('execution_lock_sha256','task_id_sha256','instance_id_sha256','recipe_sha256')) or
 not COMMIT.fullmatch(str(request.get('dataset_revision',''))) or
 not COMMIT.fullmatch(str(request.get('base_commit',''))) or
 not all(IMAGE.fullmatch(str(request.get(k,''))) for k in
 ('image_key','base_image_key','env_image_key')) or
 not PLATFORM.fullmatch(str(request.get('platform',''))) or
 any(not isinstance(request.get(k),str) or not request[k].strip() or '\x00' in request[k]
  for k in ('base_dockerfile','env_dockerfile','instance_dockerfile',
   'setup_env_script','setup_repo_script')) or
 request['recipe_sha256']!=sha(canonical(recipe).encode())):
 die('image_acquisition_request_binding_invalid')
secure_dir(ROOT,create=True)
cache=ROOT/'image-cache'; secure_dir(cache,create=True)
# Serialize by mutable image label, not by task id.  Two locale/task variants
# may resolve to the same official TestSpec image and must never race a pull,
# build, or retag operation.
image_lock=cache/('image-'+sha(request['image_key'].encode())+'.lock')
ledger_path=cache/'ledger.json'; ledger_lock=cache/'ledger.lock'
with open_private_lock(image_lock) as image_handle:
 fcntl.flock(image_handle,fcntl.LOCK_EX)
 with open_private_lock(ledger_lock) as ledger_handle:
  fcntl.flock(ledger_handle,fcntl.LOCK_EX)
  ledger=load_ledger(ledger_path,request)
  existing=ledger['entries'].get(request['task_id_sha256'])
 if existing is not None:
  entry=verify_entry(existing,request)
 else:
  # Never bless an unknown pre-existing label.  A new ledger entry must come
  # from a fresh trusted pull, or from the locked official TestSpec recipe.
  pulled=run(['podman','pull','--quiet','--policy=always',request['image_key']]).returncode==0
  digest=image_state(request['image_key']) if pulled else build(request)
  mode='pull' if pulled else 'official-recipe-build'
  if not DIGEST.fullmatch(str(digest)): die('official_image_digest_unavailable')
  image_ref=(request['image_key'].rsplit(':',1)[0]+'@'+digest) if pulled else digest
  if image_state(image_ref)!=digest: die('official_image_immutable_ref_unusable')
  entry={k:request[k] for k in ('execution_lock_sha256','dataset_revision',
   'task_id_sha256','instance_id_sha256','base_commit','image_key','recipe_sha256')}
  entry.update({'image_digest':digest,'image_ref':image_ref,'acquisition_mode':mode})
  entry['binding_sha256']=binding_hash(entry)
  with open_private_lock(ledger_lock) as ledger_handle:
   fcntl.flock(ledger_handle,fcntl.LOCK_EX)
   ledger=load_ledger(ledger_path,request)
   raced=ledger['entries'].get(request['task_id_sha256'])
   if raced is not None:
    entry=verify_entry(raced,request)
   else:
    ledger['entries'][request['task_id_sha256']]=entry
    ledger['entry_count']=len(ledger['entries'])
    ledger['content_sha256']=ledger_content(ledger)
    write_atomic(ledger_path,ledger)
result={'schema_version':BINDING_SCHEMA,**entry,
 'ledger_content_sha256':ledger['content_sha256']}
print(canonical(result))
"""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )


def _walk_objects(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _extract_session_id(stdout: str) -> str | None:
    found: set[str] = set()
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in _walk_objects(value):
            for name in ("sessionID", "session_id", "sessionId"):
                candidate = item.get(name)
                if isinstance(candidate, str) and _SESSION_ID.fullmatch(candidate):
                    found.add(candidate)
    return next(iter(found)) if len(found) == 1 else None


def load_supervisor_receipt_key(
    wsl_distro: str,
    *,
    key_path: str = SUPERVISOR_RECEIPT_KEY_PATH,
) -> bytes:
    """Read an existing root-owned 0600 key from the WSL supervisor.

    The key is never generated in a run directory, passed in argv/environment,
    or mounted into either model or evaluator containers.  Missing or rotated
    keys fail closed; an operator must provision/rotate them explicitly.
    """

    if not wsl_distro.strip() or key_path not in {
        SUPERVISOR_RECEIPT_KEY_PATH,
        DISTILLATION_RECEIPT_KEY_PATH,
    }:
        raise ExecutionContractError("v3_runtime_receipt_key_path_invalid")
    exists = subprocess.run(
        [
            "wsl.exe",
            "--distribution",
            wsl_distro,
            "--user",
            "root",
            "--exec",
            "test",
            "-f",
            key_path,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if exists.returncode != 0:
        raise ExecutionContractError("v3_runtime_receipt_key_missing")
    checked = subprocess.run(
        [
            "wsl.exe",
            "--distribution",
            wsl_distro,
            "--user",
            "root",
            "--exec",
            "stat",
            "-c",
            "%a:%u:%g:%F",
            key_path,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if checked.returncode != 0 or checked.stdout.strip() != b"600:0:0:regular file":
        raise ExecutionContractError("v3_runtime_receipt_key_permissions_invalid")
    loaded = subprocess.run(
        [
            "wsl.exe",
            "--distribution",
            wsl_distro,
            "--user",
            "root",
            "--exec",
            "cat",
            key_path,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    key = loaded.stdout
    if loaded.returncode != 0 or len(key) < 32:
        raise ExecutionContractError("v3_runtime_receipt_key_invalid")
    return key


def load_distillation_supervisor_receipt_key(wsl_distro: str) -> bytes:
    """Load the protocol-separated root-only train execution HMAC key."""

    return load_supervisor_receipt_key(
        wsl_distro,
        key_path=DISTILLATION_RECEIPT_KEY_PATH,
    )


def issue_distillation_execution_receipt_after_cleanup(
    *,
    private_root: Path,
    bindings: Mapping[str, Any],
    final_patch: bytes,
    builder_output: Mapping[str, Any],
    trusted_receipt_key: bytes,
    issued_at: str | None = None,
) -> Path:
    """Persist a self-verified receipt only after its caller completed cleanup.

    The shared evidence helper is recomputed here, so callers cannot bind a
    transcript or validation digest that disagrees with the actual builder
    artifact.  Official benchmark status is deliberately absent.
    """

    if set(bindings) != set(DISTILLATION_EXECUTION_BINDING_KEYS):
        raise ExecutionContractError("distillation_execution_bindings_invalid")
    if not final_patch or _sha256(final_patch) != bindings.get("final_patch_sha256"):
        raise ExecutionContractError("distillation_execution_final_patch_invalid")
    transcript_sha, validation_sha = distillation_tool_evidence(builder_output)
    if (
        bindings.get("tool_transcript_sha256") != transcript_sha
        or bindings.get("validation_evidence_sha256") != validation_sha
    ):
        raise ExecutionContractError("distillation_execution_evidence_unbound")
    validation_state_sha = distillation_validation_state_sha256(
        builder_output,
        final_patch_sha256=str(bindings.get("final_patch_sha256", "")),
        validator_version_sha256=str(
            bindings.get("validator_version_sha256", "")
        ),
    )
    if bindings.get("validation_state_sha256") != validation_state_sha:
        raise ExecutionContractError("distillation_execution_validation_state_unbound")
    validation_state = builder_output.get("validation_state")
    if not isinstance(validation_state, Mapping):
        raise ExecutionContractError("distillation_execution_validation_state_unbound")
    task_digest = str(bindings.get("task_id_sha256", ""))
    if not _SHA256.fullmatch(task_digest):
        raise ExecutionContractError("distillation_execution_task_id_invalid")
    timestamp = issued_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    receipt_id = _sha256(
        _canonical(
            {
                **dict(bindings),
                "issued_at": timestamp,
                "cleanup_success": True,
                "validation_state": dict(validation_state),
            }
        ).encode("utf-8")
    )
    receipt = sign_distillation_execution_receipt(
        bindings=bindings,
        validation_state=validation_state,
        receipt_id=receipt_id,
        issued_at=timestamp,
        trusted_receipt_key=trusted_receipt_key,
    )
    directory = private_root.resolve() / task_digest
    patch_path = directory / "final.patch"
    receipt_path = directory / "distillation-execution-receipt.json"
    _atomic_bytes(patch_path, final_patch)
    _atomic_json(receipt_path, receipt)
    return receipt_path


@dataclass(frozen=True)
class OfficialHarnessTask:
    instance: Mapping[str, Any]
    test_spec: object
    image_key: str


@dataclass(frozen=True)
class V3WorkspaceHandle:
    task_id: str
    instance_id: str
    base_commit: str
    image_key: str
    image_digest: str
    image_ref: str
    image_cache_binding_sha256: str
    image_acquisition_mode: str
    native_root: str
    native_testbed: str
    host_workspace: Path
    canonical_testbed: str
    materialization_id: str
    harness_task: OfficialHarnessTask

    def sandbox_contract(self) -> dict[str, str]:
        return {
            "runtime_adapter_version": RUNTIME_ADAPTER_VERSION,
            "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3,
            "instance_image_ref": self.image_ref,
            "image_cache_binding_sha256": self.image_cache_binding_sha256,
            "image_acquisition_mode": self.image_acquisition_mode,
            "native_workspace": self.native_testbed,
            "container_workspace": self.canonical_testbed,
            "network_mode": "none",
            "route_transport": "supervisor-fixed-target-unix-socket",
            "public_egress": "blocked-and-behavior-probed",
            "image_pull_policy": "never",
        }


@dataclass(frozen=True)
class OfficialEvalExecution:
    exit_code: int
    timed_out: bool
    duration_ms: float
    stdout: bytes
    stderr: bytes
    fresh_container: bool
    network_mode: str
    image_ref: str
    patch_sha256: str


@dataclass(frozen=True)
class OfficialGrade:
    resolved: bool
    report_hash: str


@dataclass(frozen=True)
class V3ModelRun:
    execution: AgentExecution
    session_export: Mapping[str, Any]
    binary_diff: bytes
    public_validation_visible: bool
    model_container_destroyed: bool


class OfficialHarnessRuntime(Protocol):
    def resolve(
        self,
        *,
        instance_id: str,
        expected_repo: str,
        expected_base_commit: str,
    ) -> OfficialHarnessTask: ...

    def grade(
        self,
        *,
        task: OfficialHarnessTask,
        patch: bytes,
        test_output: bytes,
        private_directory: Path,
    ) -> OfficialGrade: ...


class V3ContainerTransport(Protocol):
    def acquire_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding: ...

    def verify_cached_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding: ...

    def inspect_image_digest(self, image_key: str) -> str: ...

    def materialize_testbed(
        self,
        *,
        task_id: str,
        instance_id: str,
        image_ref: str,
        base_commit: str,
    ) -> tuple[str, str, Path, str]: ...

    def capture_binary_diff(self, handle: V3WorkspaceHandle) -> bytes: ...

    def workspace_inventory(self, handle: V3WorkspaceHandle) -> Mapping[str, Any]: ...

    def run_model(
        self,
        *,
        handle: V3WorkspaceHandle,
        linux_opencode: Path,
        config_bytes: bytes,
        provider_id: str,
        model_id: str,
        variant: str,
        prompt: str,
        policy: ToolPolicy,
        route_host: str,
        route_port: int,
        sample_id: str,
    ) -> V3ModelRun: ...

    def apply_binary_diff(self, handle: V3WorkspaceHandle, patch: bytes) -> None: ...

    def run_official_eval(
        self,
        *,
        handle: V3WorkspaceHandle,
        patch: bytes,
        eval_script: str,
        timeout_seconds: int,
    ) -> OfficialEvalExecution: ...

    def cleanup(self, handle: V3WorkspaceHandle) -> None: ...


class PinnedOfficialHarnessRuntime:
    """Exact-checkout TestSpec resolver and system-private official grader."""

    def __init__(self, project_root: Path, lock: Mapping[str, Any]) -> None:
        self.project_root = project_root.resolve()
        self.lock = lock
        harness = lock["official_harness"]
        dataset = lock["dataset"]
        self.checkout = (self.project_root / str(harness["checkout"])).resolve()
        self.parquet = (self.project_root / str(dataset["parquet"])).resolve()
        completed = subprocess.run(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != harness["revision"]:
            raise ExecutionContractError("v3_runtime_harness_checkout_unbound")
        if str(self.checkout) not in sys.path:
            sys.path.insert(0, str(self.checkout))
        with official_harness_import_scope():
            self.test_spec_module = importlib.import_module(str(harness["module"]))
            self.grading_module = importlib.import_module("swebench.harness.grading")

    def _instance(self, instance_id: str) -> Mapping[str, Any]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ExecutionContractError("v3_runtime_pyarrow_missing") from exc
        table = pq.read_table(
            self.parquet,
            filters=[("instance_id", "=", instance_id)],
        )
        rows = table.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ExecutionContractError("v3_runtime_instance_row_missing")
        return dict(rows[0])

    def resolve(
        self,
        *,
        instance_id: str,
        expected_repo: str,
        expected_base_commit: str,
    ) -> OfficialHarnessTask:
        instance = self._instance(instance_id)
        if (
            instance.get("repo") != expected_repo
            or instance.get("base_commit") != expected_base_commit
        ):
            raise ExecutionContractError("v3_runtime_public_private_binding_mismatch")
        image_key = resolve_official_instance_image_key(
            instance,
            self.lock,
            self.test_spec_module,
        )
        factory = getattr(
            self.test_spec_module,
            str(self.lock["official_harness"]["factory"]),
        )
        policy = self.lock["image_policy"]
        test_spec = factory(
            dict(instance),
            namespace=policy["namespace"],
            base_image_tag=policy["base_image_tag"],
            env_image_tag=policy["env_image_tag"],
            instance_image_tag=policy["instance_image_tag"],
            arch=policy["arch"],
        )
        if getattr(test_spec, "instance_image_key", None) != image_key:
            raise ExecutionContractError("v3_runtime_testspec_image_mismatch")
        return OfficialHarnessTask(
            instance=instance,
            test_spec=test_spec,
            image_key=image_key,
        )

    def grade(
        self,
        *,
        task: OfficialHarnessTask,
        patch: bytes,
        test_output: bytes,
        private_directory: Path,
    ) -> OfficialGrade:
        output_path = private_directory / "official-test-output.txt"
        report_path = private_directory / "official-report.json"
        _atomic_bytes(output_path, test_output)
        prediction = {
            "instance_id": str(task.instance["instance_id"]),
            "model_name_or_path": "anchor-v3-runtime",
            "model_patch": patch.decode("utf-8", errors="strict"),
        }
        grader = getattr(self.grading_module, "get_eval_report", None)
        if not callable(grader):
            raise ExecutionContractError("v3_runtime_official_grader_missing")
        report = grader(
            test_spec=task.test_spec,
            prediction=prediction,
            test_log_path=output_path,
            include_tests_status=True,
        )
        if not isinstance(report, Mapping):
            raise ExecutionContractError("v3_runtime_official_report_invalid")
        instance_id = str(task.instance["instance_id"])
        row = report.get(instance_id)
        if not isinstance(row, Mapping) or not isinstance(row.get("resolved"), bool):
            raise ExecutionContractError("v3_runtime_official_report_invalid")
        encoded = (_canonical(report) + "\n").encode("utf-8")
        _atomic_bytes(report_path, encoded)
        return OfficialGrade(
            resolved=bool(row["resolved"]),
            report_hash=_sha256(encoded),
        )


class WslPodmanV3Transport:
    """No-pull WSL/Podman transport for the formal runtime adapter."""

    def __init__(self, *, wsl_distro: str, native_root: str) -> None:
        if not wsl_distro.strip() or not _NATIVE_ROOT.fullmatch(native_root):
            raise ExecutionContractError("v3_runtime_transport_config_invalid")
        self.wsl_distro = wsl_distro
        self.native_root = native_root.rstrip("/")
        self._route_relays: dict[str, subprocess.Popen[bytes]] = {}

    def _run(
        self,
        arguments: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 900,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                self.wsl_distro,
                "--user",
                "root",
                "--exec",
                *arguments,
            ],
            input=input_bytes,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _image_ref(image_key: str, digest: str) -> str:
        return f"{image_key.rsplit(':', 1)[0]}@{digest}"

    def _image_cache_operation(
        self,
        request: OfficialImageAcquisitionRequest,
        *,
        acquire: bool,
    ) -> OfficialImageCacheBinding:
        request_root = (
            f"{self.native_root}/image-cache/requests/{request.task_id_sha256}"
        )
        request_path = f"{request_root}/request.json"
        self._write_native(
            request_path,
            (_canonical(request.to_dict()) + "\n").encode("utf-8"),
        )
        try:
            if acquire:
                completed = self._run(
                    [
                        "python3",
                        "-c",
                        _SUPERVISOR_IMAGE_CACHE,
                        self.native_root,
                        request_path,
                    ],
                    timeout=21600,
                )
            else:
                # The same supervisor program verifies existing bindings before
                # it considers acquisition.  For resume we must not reach its
                # pull/build branch, so first require a bound ledger entry via a
                # tiny read-only verifier.
                completed = self._verify_cached_request(request_path, request)
        finally:
            self._run(["rm", "-rf", "--", request_root], timeout=60)
        try:
            value = json.loads(completed.stdout.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionContractError(
                "v3_runtime_image_supervisor_response_invalid"
            ) from exc
        if completed.returncode != 0 or not isinstance(value, Mapping):
            raise ExecutionContractError("v3_runtime_image_supervisor_failed")
        binding = OfficialImageCacheBinding.from_mapping(value)
        if not binding.matches(request):
            raise ExecutionContractError("v3_runtime_image_cache_binding_mismatch")
        return binding

    def _verify_cached_request(
        self,
        request_path: str,
        request: OfficialImageAcquisitionRequest,
    ) -> subprocess.CompletedProcess[bytes]:
        # This verifier cannot pull or build: its subprocess allowlist contains
        # only file reads and `podman image inspect`.  It also rechecks both the
        # immutable ref and mutable label, so deletion or tag drift fails closed.
        script = r"""
import hashlib,json,pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1]); req=json.loads(pathlib.Path(sys.argv[2]).read_text())
ledger_path=root/'image-cache'/'ledger.json'
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(v): return hashlib.sha256(v).hexdigest()
try: ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
except Exception: raise SystemExit(3)
unsigned={k:ledger[k] for k in ('schema_version','execution_lock_sha256','dataset_revision','entry_count','entries')}
if (ledger.get('schema_version')!='anchor.swebench-image-cache-ledger.v1' or
 ledger.get('content_sha256')!=sha(canonical(unsigned).encode()) or
 ledger.get('execution_lock_sha256')!=req.get('execution_lock_sha256') or
 ledger.get('dataset_revision')!=req.get('dataset_revision') or
 ledger.get('entry_count')!=len(ledger.get('entries',{}))): raise SystemExit(3)
entry=ledger['entries'].get(req.get('task_id_sha256'))
if not isinstance(entry,dict): raise SystemExit(3)
keys=('execution_lock_sha256','dataset_revision','task_id_sha256','instance_id_sha256','base_commit','image_key','recipe_sha256')
if any(entry.get(k)!=req.get(k) for k in keys): raise SystemExit(3)
binding_fields={'execution_lock_sha256','dataset_revision','task_id_sha256','instance_id_sha256','base_commit','image_key','image_digest','image_ref','recipe_sha256','acquisition_mode','binding_sha256'}
if set(entry)!=binding_fields: raise SystemExit(3)
expected=sha(canonical({k:entry[k] for k in binding_fields-{'binding_sha256'}}).encode())
if entry.get('binding_sha256')!=expected: raise SystemExit(3)
def state(ref):
 p=subprocess.run(['podman','image','inspect','--format','{{.Digest}}|{{.Id}}',ref],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=60,check=False)
 if p.returncode: return None
 parts=p.stdout.decode('utf-8','replace').strip().split('|',1)
 if len(parts)!=2: return None
 pat=re.compile(r'^sha256:[0-9a-f]{64}$')
 return parts[0] if pat.fullmatch(parts[0]) else parts[1] if pat.fullmatch(parts[1]) else None
if state(entry['image_ref'])!=entry['image_digest'] or state(entry['image_key'])!=entry['image_digest']:
 raise SystemExit(3)
print(canonical({'schema_version':'anchor.swebench-image-cache-binding.v1',**entry,'ledger_content_sha256':ledger['content_sha256']}))
"""
        return self._run(
            ["python3", "-c", script, self.native_root, request_path],
            timeout=180,
        )

    def acquire_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding:
        return self._image_cache_operation(request, acquire=True)

    def verify_cached_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding:
        return self._image_cache_operation(request, acquire=False)

    def inspect_image_digest(self, image_key: str) -> str:
        if not _IMAGE_KEY.fullmatch(image_key):
            raise ExecutionContractError("v3_runtime_image_key_invalid")
        completed = self._run(
            ["podman", "image", "inspect", "--format", "{{.Digest}}", image_key],
            timeout=60,
        )
        digest = completed.stdout.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0 or not _IMAGE_DIGEST.fullmatch(digest):
            raise ExecutionContractError("v3_runtime_image_missing_or_unbound")
        return digest

    def materialize_testbed(
        self,
        *,
        task_id: str,
        instance_id: str,
        image_ref: str,
        base_commit: str,
    ) -> tuple[str, str, Path, str]:
        if (
            not _SHA256.fullmatch(task_id.rsplit(":", 1)[-1])
            or not _INSTANCE_ID.fullmatch(instance_id)
            or not (
                _IMAGE_DIGEST.fullmatch(image_ref)
                or re.fullmatch(
                    r"[a-z0-9][a-z0-9._/:\-]{1,511}@sha256:[0-9a-f]{64}",
                    image_ref,
                )
            )
            or not _COMMIT.fullmatch(base_commit)
        ):
            raise ExecutionContractError("v3_runtime_materialization_input_invalid")
        materialization_id = secrets.token_hex(16)
        native_task_root = f"{self.native_root}/live/{task_id.rsplit(':', 1)[-1]}/{materialization_id}"
        native_testbed = f"{native_task_root}/testbed"
        script = r"""
set -eu
root=$1
testbed=$2
image=$3
commit=$4
cid=''
cleanup() { [ -z "$cid" ] || podman rm -f "$cid" >/dev/null 2>&1 || true; }
trap cleanup EXIT HUP INT TERM
install -d -m 700 "$root"
[ "$(stat -f -c %T "$root")" = "ext2/ext3" ]
mkdir -m 700 "$testbed"
cid=$(podman create --pull=never --network none --read-only --cap-drop=all \
  --security-opt=no-new-privileges --pids-limit=64 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  "$image" sleep 300)
podman cp "${cid}:/testbed/." "$testbed"
[ "$(git -C "$testbed" rev-parse HEAD)" = "$commit" ]
"""
        completed = self._run(
            ["sh", "-s", "--", native_task_root, native_testbed, image_ref, base_commit],
            input_bytes=script.encode("utf-8"),
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_testbed_materialization_failed")
        relative = native_testbed.lstrip("/").replace("/", "\\")
        host_workspace = Path(f"\\\\wsl.localhost\\{self.wsl_distro}\\{relative}")
        return native_task_root, native_testbed, host_workspace, materialization_id

    def capture_binary_diff(self, handle: V3WorkspaceHandle) -> bytes:
        staged = self._run(
            ["git", "-C", handle.native_testbed, "add", "-A", "--"],
            timeout=120,
        )
        if staged.returncode != 0:
            raise ExecutionContractError("v3_runtime_diff_stage_failed")
        completed = self._run(
            [
                "git",
                "-C",
                handle.native_testbed,
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "HEAD",
                "--",
            ],
            timeout=120,
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_diff_capture_failed")
        return completed.stdout

    def workspace_inventory(self, handle: V3WorkspaceHandle) -> Mapping[str, Any]:
        completed = self._run(
            ["git", "-C", handle.native_testbed, "ls-files", "-z"],
            timeout=120,
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_workspace_inventory_failed")
        paths = [
            item.decode("utf-8", errors="replace")
            for item in completed.stdout.split(b"\0")
            if item
        ]
        return {
            "file_count": len(paths),
            "paths": paths[:5000],
            "truncated": len(paths) > 5000,
            "canonical_workspace": CANONICAL_TESTBED,
            "source": "official-testspec-instance-image",
        }

    def _copy_windows_file_to_native(self, source: Path, destination: str) -> None:
        if not source.is_file():
            raise ExecutionContractError("v3_runtime_opencode_binary_missing")
        script = 'set -eu; src=$(wslpath -u "$1"); install -D -m 500 "$src" "$2"'
        completed = self._run(
            ["sh", "-c", script, "--", str(source.resolve()), destination],
            timeout=180,
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_opencode_stage_failed")

    def _start_route_relay(
        self,
        *,
        handle: V3WorkspaceHandle,
        route_host: str,
        route_port: int,
    ) -> str:
        if (
            not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", route_host)
            or route_port < 1
            or route_port > 65535
        ):
            raise ExecutionContractError("v3_runtime_route_target_invalid")
        route_directory = f"{handle.native_root}/route"
        route_socket = f"{route_directory}/ccswitch.sock"
        prepared = self._run(["install", "-d", "-m", "700", route_directory], timeout=30)
        if prepared.returncode != 0:
            raise ExecutionContractError("v3_runtime_route_socket_stage_failed")
        process = subprocess.Popen(
            [
                "wsl.exe",
                "--distribution",
                self.wsl_distro,
                "--user",
                "root",
                "--exec",
                "python3",
                "-c",
                _HOST_UNIX_ROUTE_RELAY,
                route_socket,
                route_host,
                str(route_port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._route_relays[handle.materialization_id] = process
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            checked = self._run(["test", "-S", route_socket], timeout=5)
            if checked.returncode == 0:
                return route_directory
            time.sleep(0.1)
        self._stop_route_relay(handle)
        raise ExecutionContractError("v3_runtime_route_socket_start_failed")

    def _stop_route_relay(self, handle: V3WorkspaceHandle) -> None:
        process = self._route_relays.pop(handle.materialization_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def run_model(
        self,
        *,
        handle: V3WorkspaceHandle,
        linux_opencode: Path,
        config_bytes: bytes,
        provider_id: str,
        model_id: str,
        variant: str,
        prompt: str,
        policy: ToolPolicy,
        route_host: str,
        route_port: int,
        sample_id: str,
    ) -> V3ModelRun:
        if (
            not config_bytes
            or not provider_id.strip()
            or not model_id.strip()
            or variant != "max"
            or not prompt.strip()
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", sample_id)
        ):
            raise ExecutionContractError("v3_runtime_model_input_invalid")
        private_native = f"{handle.native_root}/model-private"
        binary_native = f"{private_native}/opencode"
        config_native = f"{private_native}/opencode.json"
        self._copy_windows_file_to_native(linux_opencode, binary_native)
        self._write_native(config_native, config_bytes)
        route_directory = self._start_route_relay(
            handle=handle,
            route_host=route_host,
            route_port=route_port,
        )
        container_id = ""
        stdout = b""
        stderr = b""
        export: Mapping[str, Any] | None = None
        started = time.perf_counter()
        timed_out = False
        return_code = 127
        model_destroyed = False
        try:
            created = self._run(
                [
                    "podman",
                    "create",
                    "--pull=never",
                    "--network=none",
                    "--workdir=/testbed",
                    "--read-only",
                    "--cap-drop=all",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=512",
                    "--memory=8g",
                    "--cpus=4",
                    "--tmpfs=/tmp:rw,nosuid,size=1g",
                    "--tmpfs=/anchor/state:rw,noexec,nosuid,size=1g",
                    "--mount",
                    f"type=bind,src={handle.native_testbed},dst=/testbed,rw",
                    "--mount",
                    f"type=bind,src={binary_native},dst=/anchor/bin/opencode,ro",
                    "--mount",
                    f"type=bind,src={config_native},dst=/anchor/config/opencode.json,ro",
                    "--mount",
                    f"type=bind,src={route_directory},dst=/run/anchor-route,ro",
                    handle.image_ref,
                    "sleep",
                    str(policy.timeout_seconds + 180),
                ],
                timeout=120,
            )
            container_id = created.stdout.decode("utf-8", errors="replace").strip()
            if created.returncode != 0 or not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
                raise ExecutionContractError("v3_runtime_model_container_create_failed")
            started_container = self._run(["podman", "start", container_id], timeout=60)
            if started_container.returncode != 0:
                raise ExecutionContractError("v3_runtime_model_container_start_failed")
            bridge = self._run(
                [
                    "podman",
                    "exec",
                    "-d",
                    container_id,
                    "python3",
                    "-c",
                    _CONTAINER_TCP_UNIX_BRIDGE,
                    "/run/anchor-route/ccswitch.sock",
                    "18080",
                ],
                timeout=30,
            )
            if bridge.returncode != 0:
                raise ExecutionContractError("v3_runtime_container_route_bridge_failed")
            isolation_probe = r"""
import socket,sys,urllib.request
for path in ('/anchor/health','/v1/models'):
 r=urllib.request.urlopen('http://127.0.0.1:18080'+path,timeout=10)
 assert r.status==200
for host,port in [('1.1.1.1',443),('169.254.169.254',80),(sys.argv[1],80)]:
 try: socket.create_connection((host,port),timeout=1)
 except OSError: pass
 else: raise SystemExit(20)
try: socket.getaddrinfo('github.com',443)
except OSError: pass
else: raise SystemExit(21)
"""
            probe = self._run(
                [
                    "podman",
                    "exec",
                    container_id,
                    "python3",
                    "-c",
                    isolation_probe,
                    route_host,
                ],
                timeout=45,
            )
            if probe.returncode != 0:
                raise ExecutionContractError("v3_runtime_model_egress_probe_failed")
            environment = [
                "-e",
                "OPENCODE_CONFIG=/anchor/config/opencode.json",
                "-e",
                "OPENCODE_CONFIG_DIR=/anchor/state/config",
                "-e",
                "XDG_CONFIG_HOME=/anchor/state/config",
                "-e",
                "XDG_DATA_HOME=/anchor/state/data",
                "-e",
                "XDG_CACHE_HOME=/anchor/state/cache",
                "-e",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS=true",
                "-e",
                "OPENCODE_DISABLE_LSP_DOWNLOAD=true",
                "-e",
                "OPENCODE_DISABLE_MODELS_FETCH=true",
                "-e",
                "ANCHOR_LOCAL_ROUTE_CLIENT_TOKEN=anchor-local-route",
            ]
            command = [
                "podman",
                "exec",
                *environment,
                "--workdir=/testbed",
                container_id,
                "/anchor/bin/opencode",
                "run",
                "--format=json",
                "--model",
                f"{provider_id}/{model_id}",
                "--agent=anchor-distiller",
                "--variant",
                variant,
                "--title",
                f"anchor-distiller:{sample_id}",
                prompt,
            ]
            try:
                completed = self._run(command, timeout=policy.timeout_seconds)
                return_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                return_code = 124
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
            session_id = _extract_session_id(stdout.decode("utf-8", errors="replace"))
            if session_id is not None:
                exported = self._run(
                    [
                        "podman",
                        "exec",
                        *environment,
                        container_id,
                        "/anchor/bin/opencode",
                        "export",
                        session_id,
                    ],
                    timeout=60,
                )
                if exported.returncode == 0:
                    try:
                        parsed = json.loads(exported.stdout.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, Mapping):
                        export = parsed
        finally:
            if container_id:
                removed = self._run(["podman", "rm", "-f", container_id], timeout=60)
                model_destroyed = removed.returncode == 0
            self._stop_route_relay(handle)
        if export is None or not model_destroyed:
            raise ExecutionContractError("v3_runtime_model_export_or_cleanup_failed")
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        trace, rejected = parse_opencode_jsonl(stdout_text, policy)
        errors = list(classify_error_metadata(stdout_text, stderr_text))
        if timed_out:
            errors.append("wrapper_timeout")
        execution = AgentExecution(
            exit_code=return_code,
            timed_out=timed_out,
            duration_ms=(time.perf_counter() - started) * 1000,
            trace=trace,
            stdout_sha256=digest_text(stdout_text),
            stderr_sha256=digest_text(stderr_text),
            rejected_events=rejected,
            error_codes=tuple(dict.fromkeys(errors)),
            public_outcome=parse_public_outcome(stdout_text),
            controlled_session_id=_extract_session_id(stdout_text),
            opencode_version=None,
        )
        binary_diff = self.capture_binary_diff(handle)
        public_validation_visible = any(
            item.source == "agent"
            and item.tool == "bash"
            and item.status == "completed"
            and item.exit_code == 0
            for item in trace
        )
        return V3ModelRun(
            execution=execution,
            session_export=export,
            binary_diff=binary_diff,
            public_validation_visible=public_validation_visible,
            model_container_destroyed=model_destroyed,
        )

    def apply_binary_diff(self, handle: V3WorkspaceHandle, patch: bytes) -> None:
        if not patch or b"\x00" in patch:
            raise ExecutionContractError("v3_runtime_resume_patch_invalid")
        for extra in (["--check"], []):
            completed = self._run(
                [
                    "git",
                    "-C",
                    handle.native_testbed,
                    "apply",
                    "--binary",
                    "--index",
                    "--whitespace=nowarn",
                    *extra,
                    "-",
                ],
                input_bytes=patch,
                timeout=120,
            )
            if completed.returncode != 0:
                raise ExecutionContractError("v3_runtime_resume_patch_rejected")

    def _write_native(self, path: str, value: bytes) -> None:
        writer = r"""
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1])
root=pathlib.Path(sys.argv[2])
assert p.is_absolute() and root.is_absolute() and p!=root
relative=p.relative_to(root)
current=root
if not current.exists(): current.mkdir(mode=0o700)
for part in relative.parts[:-1]:
 s=current.lstat()
 assert stat.S_ISDIR(s.st_mode) and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o700
 current=current/part
 if not current.exists(): current.mkdir(mode=0o700)
s=current.lstat()
assert current==p.parent and stat.S_ISDIR(s.st_mode) and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o700
t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
if hasattr(os,'O_NOFOLLOW'): flags|=os.O_NOFOLLOW
fd=os.open(t,flags,0o600)
try:
 with os.fdopen(fd,'wb',closefd=False) as handle:
  handle.write(sys.stdin.buffer.read()); handle.flush(); os.fsync(handle.fileno())
finally:
 try: os.close(fd)
 except OSError: pass
os.replace(t,p); os.chmod(p,0o600)
"""
        completed = self._run(
            [
                "python3",
                "-c",
                writer,
                path,
                self.native_root,
            ],
            input_bytes=value,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_private_stage_failed")

    def run_official_eval(
        self,
        *,
        handle: V3WorkspaceHandle,
        patch: bytes,
        eval_script: str,
        timeout_seconds: int,
    ) -> OfficialEvalExecution:
        if not patch or not eval_script.strip() or timeout_seconds < 1:
            raise ExecutionContractError("v3_runtime_official_eval_input_invalid")
        private_native = f"{handle.native_root}/official-eval"
        self._write_native(f"{private_native}/final.patch", patch)
        self._write_native(f"{private_native}/eval.sh", eval_script.encode("utf-8"))
        command = [
            "podman",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--workdir=/testbed",
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            "--memory=8g",
            "--cpus=4",
            "--tmpfs=/tmp:rw,nosuid,size=1g",
            "--mount",
            f"type=bind,src={private_native},dst=/anchor/private,ro",
            handle.image_ref,
            "bash",
            "-lc",
            "git apply --binary --whitespace=nowarn /anchor/private/final.patch || exit 125; exec bash /anchor/private/eval.sh",
        ]
        started = time.perf_counter()
        try:
            completed = self._run(command, timeout=timeout_seconds + 30)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(
                command,
                124,
                stdout=exc.stdout or b"",
                stderr=exc.stderr or b"",
            )
            timed_out = True
        duration_ms = (time.perf_counter() - started) * 1000
        if timed_out or completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_official_eval_failed")
        return OfficialEvalExecution(
            exit_code=completed.returncode,
            timed_out=False,
            duration_ms=duration_ms,
            stdout=completed.stdout,
            stderr=completed.stderr,
            fresh_container=True,
            network_mode="none",
            image_ref=handle.image_ref,
            patch_sha256=_sha256(patch),
        )

    def cleanup(self, handle: V3WorkspaceHandle) -> None:
        expected_prefix = f"{self.native_root}/live/"
        if not handle.native_root.startswith(expected_prefix):
            raise ExecutionContractError("v3_runtime_cleanup_scope_invalid")
        completed = self._run(
            ["rm", "-rf", "--", handle.native_root],
            timeout=120,
        )
        if completed.returncode != 0:
            raise ExecutionContractError("v3_runtime_cleanup_failed")


class SWEbenchV3RuntimeAdapter:
    """High-level, fail-closed runtime used directly by formal LiveBackend."""

    def __init__(
        self,
        *,
        project_root: Path,
        lock_path: Path,
        expected_lock_sha256: str,
        private_root: Path,
        official_eval_timeout_seconds: int,
        harness: OfficialHarnessRuntime,
        transport: V3ContainerTransport,
        receipt_key: bytes,
    ) -> None:
        self.project_root = project_root.resolve()
        self.lock_path = lock_path.resolve()
        self.lock_sha256 = expected_lock_sha256
        self.lock = load_execution_lock(
            self.project_root,
            self.lock_path,
            expected_sha256=expected_lock_sha256,
        )
        self.private_root = private_root.resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        if official_eval_timeout_seconds < 1 or len(receipt_key) < 32:
            raise ExecutionContractError("v3_runtime_adapter_config_invalid")
        self.official_eval_timeout_seconds = official_eval_timeout_seconds
        self.harness = harness
        self.transport = transport
        self.receipt_key = receipt_key
        self.key_id = _sha256(receipt_key)[:16]

    @classmethod
    def production(
        cls,
        *,
        project_root: Path,
        lock_path: Path,
        expected_lock_sha256: str,
        private_root: Path,
        official_eval_timeout_seconds: int,
    ) -> "SWEbenchV3RuntimeAdapter":
        lock = load_execution_lock(
            project_root.resolve(),
            lock_path.resolve(),
            expected_sha256=expected_lock_sha256,
        )
        runtime = lock["runtime"]
        return cls(
            project_root=project_root,
            lock_path=lock_path,
            expected_lock_sha256=expected_lock_sha256,
            private_root=private_root,
            official_eval_timeout_seconds=official_eval_timeout_seconds,
            harness=PinnedOfficialHarnessRuntime(project_root, lock),
            transport=WslPodmanV3Transport(
                wsl_distro=str(runtime["wsl_distro"]),
                native_root=str(runtime["native_probe_root"]),
            ),
            receipt_key=load_supervisor_receipt_key(str(runtime["wsl_distro"])),
        )

    @staticmethod
    def _source(task: Mapping[str, Any]) -> tuple[str, str, str]:
        source = task.get("source")
        if not isinstance(source, Mapping):
            raise ExecutionContractError("v3_runtime_task_source_invalid")
        instance_id = source.get("instance_id")
        repo = source.get("repo")
        base_commit = source.get("base_commit")
        if (
            not isinstance(instance_id, str)
            or not _INSTANCE_ID.fullmatch(instance_id)
            or not isinstance(repo, str)
            or not repo.strip()
            or not isinstance(base_commit, str)
            or not _COMMIT.fullmatch(base_commit)
        ):
            raise ExecutionContractError("v3_runtime_task_source_invalid")
        return instance_id, repo, base_commit

    def _image_request(
        self,
        *,
        task_id: str,
        instance_id: str,
        base_commit: str,
        harness_task: OfficialHarnessTask,
    ) -> OfficialImageAcquisitionRequest:
        dataset = self.lock["dataset"]
        return OfficialImageAcquisitionRequest.from_test_spec(
            execution_lock_sha256=self.lock_sha256,
            dataset_revision=str(dataset["revision"]),
            task_id=task_id,
            instance_id=instance_id,
            base_commit=base_commit,
            image_key=harness_task.image_key,
            test_spec=harness_task.test_spec,
        )

    def prepare_task(self, task_id: str, task: Mapping[str, Any]) -> V3WorkspaceHandle:
        if not _SHA256.fullmatch(task_id.rsplit(":", 1)[-1]):
            raise ExecutionContractError("v3_runtime_task_id_invalid")
        instance_id, repo, base_commit = self._source(task)
        harness_task = self.harness.resolve(
            instance_id=instance_id,
            expected_repo=repo,
            expected_base_commit=base_commit,
        )
        image_request = self._image_request(
            task_id=task_id,
            instance_id=instance_id,
            base_commit=base_commit,
            harness_task=harness_task,
        )
        binding = self.transport.acquire_official_image(image_request)
        image_digest = binding.image_digest
        image_ref = binding.image_ref
        native_root, native_testbed, host_workspace, materialization_id = (
            self.transport.materialize_testbed(
                task_id=task_id,
                instance_id=instance_id,
                image_ref=image_ref,
                base_commit=base_commit,
            )
        )
        return V3WorkspaceHandle(
            task_id=task_id,
            instance_id=instance_id,
            base_commit=base_commit,
            image_key=harness_task.image_key,
            image_digest=image_digest,
            image_ref=image_ref,
            image_cache_binding_sha256=binding.binding_sha256,
            image_acquisition_mode=binding.acquisition_mode,
            native_root=native_root,
            native_testbed=native_testbed,
            host_workspace=host_workspace,
            canonical_testbed=CANONICAL_TESTBED,
            materialization_id=materialization_id,
            harness_task=harness_task,
        )

    def capture_binary_diff(self, handle: V3WorkspaceHandle) -> bytes:
        return self.transport.capture_binary_diff(handle)

    def workspace_inventory(self, handle: V3WorkspaceHandle) -> Mapping[str, Any]:
        return self.transport.workspace_inventory(handle)

    def run_model(
        self,
        *,
        handle: V3WorkspaceHandle,
        linux_opencode: Path,
        config_bytes: bytes,
        provider_id: str,
        model_id: str,
        variant: str,
        prompt: str,
        policy: ToolPolicy,
        route_host: str,
        route_port: int,
        sample_id: str,
    ) -> V3ModelRun:
        return self.transport.run_model(
            handle=handle,
            linux_opencode=linux_opencode,
            config_bytes=config_bytes,
            provider_id=provider_id,
            model_id=model_id,
            variant=variant,
            prompt=prompt,
            policy=policy,
            route_host=route_host,
            route_port=route_port,
            sample_id=sample_id,
        )

    def restore_binary_diff(self, handle: V3WorkspaceHandle, patch: bytes) -> None:
        self.transport.apply_binary_diff(handle, patch)

    def _private_directory(self, task_id: str) -> Path:
        return self.private_root / _sha256(task_id.encode("utf-8"))

    def finalization_outcome(
        self,
        task_id: str,
        task: Mapping[str, Any],
        *,
        checkpoint_id: str,
        revision: int,
    ) -> str | None:
        instance_id, repo, base_commit = self._source(task)
        directory = self._private_directory(task_id)
        receipt_path = directory / "official-eval-receipt.json"
        patch_path = directory / "final.patch"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            patch = patch_path.read_bytes()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(receipt, Mapping):
            return None
        try:
            harness_task = self.harness.resolve(
                instance_id=instance_id,
                expected_repo=repo,
                expected_base_commit=base_commit,
            )
            image_request = self._image_request(
                task_id=task_id,
                instance_id=instance_id,
                base_commit=base_commit,
                harness_task=harness_task,
            )
            binding = self.transport.verify_cached_official_image(image_request)
            image_digest = binding.image_digest
        except (OSError, ExecutionContractError):
            return None
        expected_bindings = {
            "checkpoint_id": checkpoint_id,
            "task_id_sha256": _sha256(task_id.encode("utf-8")),
            "revision": revision,
            "instance_id_sha256": _sha256(instance_id.encode("utf-8")),
            "image_digest": image_digest,
            "base_commit": base_commit,
            "patch_sha256": _sha256(patch),
            "lock_sha256": self.lock_sha256,
        }
        if not verify_official_eval_receipt(
            receipt,
            trusted_receipt_key=self.receipt_key,
            expected_bindings=expected_bindings,
            require_pass=False,
        ):
            return None
        return "completed" if receipt.get("status") == "PASS" else "failed"

    def finalize(
        self,
        *,
        handle: V3WorkspaceHandle,
        expected_cumulative_diff: bytes,
        checkpoint_id: str,
        revision: int,
    ) -> Mapping[str, Any]:
        observed = self.capture_binary_diff(handle)
        if not observed or observed != expected_cumulative_diff:
            raise ExecutionContractError("v3_runtime_final_diff_binding_failed")
        private_directory = self._private_directory(handle.task_id)
        patch_path = private_directory / "final.patch"
        receipt_path = private_directory / "official-eval-receipt.json"
        _atomic_bytes(patch_path, observed)
        eval_script = getattr(handle.harness_task.test_spec, "eval_script", None)
        if not isinstance(eval_script, str) or not eval_script.strip():
            raise ExecutionContractError("v3_runtime_official_eval_script_missing")
        image_request = self._image_request(
            task_id=handle.task_id,
            instance_id=handle.instance_id,
            base_commit=handle.base_commit,
            harness_task=handle.harness_task,
        )
        cache_binding = self.transport.verify_cached_official_image(image_request)
        if (
            cache_binding.image_digest != handle.image_digest
            or cache_binding.image_ref != handle.image_ref
            or cache_binding.binding_sha256 != handle.image_cache_binding_sha256
        ):
            raise ExecutionContractError("v3_runtime_image_cache_changed_before_eval")
        execution = self.transport.run_official_eval(
            handle=handle,
            patch=observed,
            eval_script=eval_script,
            timeout_seconds=self.official_eval_timeout_seconds,
        )
        if (
            not execution.fresh_container
            or execution.network_mode != "none"
            or execution.image_ref != handle.image_ref
            or execution.patch_sha256 != _sha256(observed)
            or execution.timed_out
            or execution.exit_code != 0
        ):
            raise ExecutionContractError("v3_runtime_official_eval_isolation_failed")
        grade = self.harness.grade(
            task=handle.harness_task,
            patch=observed,
            test_output=execution.stdout,
            private_directory=private_directory,
        )
        bindings = {
            "checkpoint_id": checkpoint_id,
            "task_id_sha256": _sha256(handle.task_id.encode("utf-8")),
            "revision": revision,
            "instance_id_sha256": _sha256(handle.instance_id.encode("utf-8")),
            "image_digest": handle.image_digest,
            "base_commit": handle.base_commit,
            "patch_sha256": _sha256(observed),
            "lock_sha256": self.lock_sha256,
        }
        receipt_id = _sha256(
            _canonical(
                {
                    **bindings,
                    "materialization_id": handle.materialization_id,
                    "report_hash": grade.report_hash,
                }
            ).encode("utf-8")
        )
        receipt = sign_official_eval_receipt(
            bindings=bindings,
            receipt_id=receipt_id,
            key_id=self.key_id,
            status="PASS" if grade.resolved else "FAIL",
            exit_code=execution.exit_code,
            duration_ms=execution.duration_ms,
            stdout_sha256=_sha256(execution.stdout),
            stderr_sha256=_sha256(execution.stderr),
            report_hash=grade.report_hash,
            trusted_receipt_key=self.receipt_key,
        )
        _atomic_json(receipt_path, receipt)
        probe_binding = {
            "schema_version": "anchor.swebench-representative-runtime-binding.v1",
            "checkpoint_id": checkpoint_id,
            "task_id_sha256": bindings["task_id_sha256"],
            "revision": revision,
            "instance_id_sha256": bindings["instance_id_sha256"],
            "image_key_sha256": _sha256(handle.image_key.encode("utf-8")),
            "image_digest": handle.image_digest,
            "image_cache_binding_sha256": handle.image_cache_binding_sha256,
            "base_commit": handle.base_commit,
            "final_patch_sha256": bindings["patch_sha256"],
            "official_receipt_sha256": _sha256(receipt_path.read_bytes()),
            "lock_sha256": self.lock_sha256,
            "content_free": True,
        }
        probe_binding["content_sha256"] = _sha256(
            _canonical(probe_binding).encode("utf-8")
        )
        probe_binding_path = private_directory / "representative-runtime-binding.json"
        _atomic_json(probe_binding_path, probe_binding)
        return {
            "schema_version": "anchor.swebench-runtime-finalization.v3",
            "completed": True,
            "gold_eligible": grade.resolved,
            "patch_sha256": bindings["patch_sha256"],
            "receipt_sha256": _sha256(receipt_path.read_bytes()),
            "representative_runtime_binding_sha256": _sha256(
                probe_binding_path.read_bytes()
            ),
            "content_free": True,
        }

    def cleanup(self, handle: V3WorkspaceHandle) -> None:
        self.transport.cleanup(handle)
