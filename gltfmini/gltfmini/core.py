"""gltfmini 核心 API：最小 glTF 2.0 校验器。

设计要点：
- 错误带 JSON Pointer（RFC 6901），并用 code 区分 MISSING（缺字段）、
  TYPE（类型错误）、VALUE（取值非法）三类。
- Gltf.validate 只使用局部变量、不写实例/全局状态，多线程并发安全。
- 错误路径才构造字符串，热路径仅做字典/列表遍历，满足 10 万次 < 500 ms。
"""

import math

# glTF 2.0 accessor.type 合法取值
_SCALAR_TYPES = ("SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4")

# glTF 2.0 accessor.componentType 合法取值
_COMPONENT_TYPES = frozenset((5120, 5121, 5122, 5123, 5125, 5126))

# 错误码：缺字段 / 类型错误 / 取值错误
MISSING = "MISSING"
TYPE = "TYPE"
VALUE = "VALUE"

__all__ = ("Gltf", "GltfError", "MISSING", "TYPE", "VALUE")


class GltfError(str):
    """带结构化字段的校验错误；本身仍是字符串，包含 JSON Pointer。"""

    def __new__(cls, pointer, code, message):
        self = super().__new__(cls, pointer + ": " + code + ": " + message)
        self.pointer = pointer
        self.code = code
        self.message = message
        return self


def _is_int(value):
    """glTF 整数字段：bool 不是合法整数。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _type_name(value):
    name = type(value).__name__
    if isinstance(value, float) and not math.isfinite(value):
        return name + " (non-finite)"
    return name


def _escape(token):
    """JSON Pointer 转义（RFC 6901）。"""
    return str(token).replace("~", "~0").replace("/", "~1")


class Gltf:
    """无状态的最小 glTF 校验器，可被多线程共享并发调用。"""

    def validate(self, doc):
        """校验已解析的 glTF 文档，返回 GltfError 列表（空列表表示通过）。

        顶层不是对象时抛 ValueError；其余问题全部收集进返回值。
        """
        if not isinstance(doc, dict):
            raise ValueError(
                "glTF document must be a JSON object, got " + type(doc).__name__
            )

        errors = []
        add = errors.append
        scalar_types = _SCALAR_TYPES
        component_types = _COMPONENT_TYPES

        # ---- asset / asset.version（必填） ----
        asset = doc.get("asset")
        if not isinstance(asset, dict):
            if asset is None and "asset" not in doc:
                add(GltfError("/asset", MISSING, "required field 'asset' is missing"))
            else:
                add(GltfError(
                    "/asset", TYPE, "expected object, got " + _type_name(asset)
                ))
        else:
            if "version" not in asset:
                add(GltfError(
                    "/asset/version", MISSING,
                    "required field 'version' is missing",
                ))
            else:
                version = asset["version"]
                if not isinstance(version, str):
                    add(GltfError(
                        "/asset/version", TYPE,
                        "expected string, got " + _type_name(version),
                    ))
                elif version != "2.0":
                    add(GltfError(
                        "/asset/version", VALUE,
                        "unsupported version " + repr(version),
                    ))

        # ---- buffers[].byteLength（非负整数） ----
        if "buffers" in doc:
            buffers = doc["buffers"]
            if not isinstance(buffers, list):
                add(GltfError(
                    "/buffers", TYPE, "expected array, got " + _type_name(buffers)
                ))
            else:
                for i, buf in enumerate(buffers):
                    pointer = "/buffers/" + str(i)
                    if not isinstance(buf, dict):
                        add(GltfError(
                            pointer, TYPE,
                            "expected object, got " + _type_name(buf),
                        ))
                        continue
                    field = pointer + "/byteLength"
                    if "byteLength" not in buf:
                        add(GltfError(
                            field, MISSING,
                            "required field 'byteLength' is missing",
                        ))
                    else:
                        byte_length = buf["byteLength"]
                        if not _is_int(byte_length):
                            add(GltfError(
                                field, TYPE,
                                "expected non-negative integer, got "
                                + _type_name(byte_length),
                            ))
                        elif byte_length < 0:
                            add(GltfError(
                                field, VALUE,
                                "byteLength must be >= 0, got " + str(byte_length),
                            ))

        # ---- accessors[].count / type / componentType ----
        if "accessors" in doc:
            accessors = doc["accessors"]
            if not isinstance(accessors, list):
                add(GltfError(
                    "/accessors", TYPE,
                    "expected array, got " + _type_name(accessors),
                ))
            else:
                for i, accessor in enumerate(accessors):
                    pointer = "/accessors/" + str(i)
                    if not isinstance(accessor, dict):
                        add(GltfError(
                            pointer, TYPE,
                            "expected object, got " + _type_name(accessor),
                        ))
                        continue

                    # count：正整数
                    count_field = pointer + "/count"
                    if "count" not in accessor:
                        add(GltfError(
                            count_field, MISSING,
                            "required field 'count' is missing",
                        ))
                    else:
                        count = accessor["count"]
                        if not _is_int(count):
                            add(GltfError(
                                count_field, TYPE,
                                "expected positive integer, got " + _type_name(count),
                            ))
                        elif count < 1:
                            add(GltfError(
                                count_field, VALUE,
                                "count must be >= 1, got " + str(count),
                            ))

                    # type：枚举字符串
                    type_field = pointer + "/type"
                    if "type" not in accessor:
                        add(GltfError(
                            type_field, MISSING,
                            "required field 'type' is missing",
                        ))
                    else:
                        accessor_type = accessor["type"]
                        if not isinstance(accessor_type, str):
                            add(GltfError(
                                type_field, TYPE,
                                "expected string, got " + _type_name(accessor_type),
                            ))
                        elif accessor_type not in scalar_types:
                            add(GltfError(
                                type_field, VALUE,
                                "invalid accessor type " + repr(accessor_type),
                            ))

                    # componentType：枚举整数
                    ct_field = pointer + "/componentType"
                    if "componentType" not in accessor:
                        add(GltfError(
                            ct_field, MISSING,
                            "required field 'componentType' is missing",
                        ))
                    else:
                        component_type = accessor["componentType"]
                        if not _is_int(component_type):
                            add(GltfError(
                                ct_field, TYPE,
                                "expected integer enum, got "
                                + _type_name(component_type),
                            ))
                        elif component_type not in component_types:
                            add(GltfError(
                                ct_field, VALUE,
                                "invalid componentType " + str(component_type),
                            ))

        return errors
