public String toString(List<T> objList) {
  if (objList == null || objList.isEmpty()) { return DEFAULT_TO_STRING_VALUE; }
  return objList.stream().map(obj ->
    obj.toString()).collect(Collectors.joining(","));
}
