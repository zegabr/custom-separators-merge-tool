public String toString(List<T> objList) {
  if (objList.size() == 0) { return DEFAULT_TO_STRING_VALUE; }
  return objList.stream().map(obj ->
    obj.toString()).collect(Collectors.joining(","));
}
